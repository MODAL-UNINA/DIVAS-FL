# %% 
import os
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

import random
import argparse
import json
from scipy.io.wavfile import write
import module.commons as commons
import utils

from augmentation.aug import Augment
from model.DIVAS import Wav2vec2, DIVAS
from utils.data_loader import AudioDataset, MelSpectrogramFixed
from utils import utils
from vocoder.hifigan import HiFi
from vocoder.bigvgan import BigvGAN
import types
import numpy as np
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic 

import logging
import warnings

logging.basicConfig(level=logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


torch.backends.cudnn.benchmark = True
global_step = 0

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def get_hparams_from_dict2(config_dict):
    return types.SimpleNamespace(**config_dict)

def main(hps):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    port = 50000 + random.randint(0, 100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))

def get_yaapt_f0(audio, sr=16000, interp=False):
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0) 
        pitch = pYAAPT.yaapt(basic.SignalObj(y_pad, sr), 
                             **{'frame_length': 20.0, 'frame_space': 5.0, 'nccf_thresh1': 0.25, 'tda_frame_length': 25.0})
        f0s.append(pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :])

    return np.vstack(f0s)  

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda(rank)

    train_dataset = AudioDataset(hps, training=True)
    train_sampler = DistributedSampler(train_dataset) if n_gpus > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=hps.train.batch_size, num_workers=4,
        sampler=train_sampler, drop_last=True, persistent_workers=True, pin_memory=True
    )
    if rank == 0:
        test_dataset = AudioDataset(hps, training=False)
        eval_loader = DataLoader(test_dataset, batch_size=1)

    w2v = Wav2vec2().cuda(rank)
    aug = Augment(hps).cuda(rank)

    model = DIVAS(hps.data.n_mel_channels, hps.diffusion.spk_dim,
                hps.diffusion.dec_dim, hps.diffusion.beta_min, hps.diffusion.beta_max, hps).cuda()

    if hps.vocoder.voc == "bigvgan":
        net_v = BigvGAN(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **vars(hps.model)
        ).cuda()
        path_ckpt = './vocoder/voc_bigvgan.pth'
    elif hps.vocoder.voc == "hifigan":
        net_v = HiFi(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **vars(hps.model)
            ).cuda()
        path_ckpt = './vocoder/voc_hifigan.pth'

    utils.load_checkpoint(path_ckpt, net_v, None)
    net_v.eval()
    net_v.dec.remove_weight_norm()

    if rank == 0:
        num_param = get_param_num(model.encoder)
        print('[Encoder] number of Parameters:', num_param)
        num_param = get_param_num(model.f0_dec)
        print('[F0 Decoder] number of Parameters:', num_param)
        num_param = get_param_num(model.mel_dec)
        print('[Mel Decoder] number of Parameters:', num_param)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    model = DDP(model, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), model, optimizer)
        global_step = (epoch_str - 1) * len(train_loader)
        print(f"Resuming from epoch {epoch_str}, global step {global_step}")
    except:
        epoch_str = 1
        global_step = 0
        print("No checkpoint found, starting from scratch.")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scaler = GradScaler('cuda', enabled=hps.train.fp16_run)

    print("Starting training...")
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [model, mel_fn, w2v, aug, net_v], optimizer,
                            scheduler_g, scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [model, mel_fn, w2v, aug, net_v], optimizer,
                            scheduler_g, scaler, [train_loader, None], None, None)
        scheduler_g.step()

    utils.save_checkpoint(model, optimizer, hps.train.learning_rate, epoch,
        os.path.join(hps.model_dir, "G_final.pth"))

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    model, mel_fn, w2v, aug, net_v = nets
    optimizer = optims
    scheduler_g = schedulers
    train_loader, eval_loader = loaders

    if writers is not None:
        writer, writer_eval = writers
    global global_step

    total_loss_mel_diff = 0
    total_loss_f0_diff = 0
    total_loss_vae = 0
    num_batches = 0
    
    train_loader.sampler.set_epoch(epoch)
    model.train()
    fixed_noise = torch.randn(1, hps.model.latent_dim).cuda(rank)
    for batch_idx, (x, norm_f0, x_f0, length, gender) in enumerate(train_loader):
        x = x.cuda(rank, non_blocking=True)
        norm_f0 = norm_f0.cuda(rank, non_blocking=True)
        x_f0 = x_f0.cuda(rank, non_blocking=True)
        length = length.cuda(rank, non_blocking=True).squeeze()
        gender = gender.cuda(rank, non_blocking=True).squeeze()

        mel_x = mel_fn(x)
        aug_x = aug(x)
        nan_x = torch.isnan(aug_x).any()
        x = x if nan_x else aug_x
        x_pad = F.pad(x, (40, 40), "reflect")
        
        w2v_x = w2v(x_pad)
        f0_x = torch.tensor(x_f0, dtype=torch.float32)

        optimizer.zero_grad()
        loss_mel_diff, loss_mel_diff_rec, loss_f0_diff, loss_mel, loss_f0, loss_vae = model.module.compute_loss(mel_x, w2v_x, norm_f0, f0_x, length, gender)
        loss_gen_all = loss_mel_diff + loss_mel_diff_rec + loss_f0_diff + loss_mel*hps.train.c_mel + loss_f0 + loss_vae

        if hps.train.fp16_run:
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optimizer)
            grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_gen_all.backward()
            grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
            optimizer.step()

        total_loss_mel_diff += loss_mel_diff.item()
        total_loss_f0_diff += loss_f0_diff.item()
        total_loss_vae += loss_vae.item()
        num_batches += 1
    
    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))
        avg_loss_mel_diff = total_loss_mel_diff / num_batches
        avg_loss_f0_diff = total_loss_f0_diff / num_batches
        avg_loss_vae = total_loss_vae / num_batches
        logger.info(f"Epoch {epoch} average losses - mel_diff: {avg_loss_mel_diff:.4f}, f0_diff: {avg_loss_f0_diff:.4f}, vae: {avg_loss_vae:.4f}")

        if epoch % hps.train.eval_interval == 0:
            # Valutazione e early stopping
            mel_loss, enc_loss, enc_f0_loss, diff_f0_loss, vae_loss = evaluate(hps, model, mel_fn, w2v, net_v, eval_loader, writer_eval, fixed_noise)
            logger.info(f"Validation losses - mel: {mel_loss:.4f}, enc_mel: {enc_loss:.4f}, enc_f0: {enc_f0_loss:.4f}, diff_f0: {diff_f0_loss:.4f}, vae: {vae_loss:.4f}")
            utils.save_checkpoint(model, optimizer, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, "G_best.pth"))


def evaluate(hps, model, mel_fn, w2v, net_v, eval_loader, writer_eval, fixed_noise):
    model.eval()
    audio_dict = {}
    mel_loss = 0
    enc_loss = 0
    enc_f0_loss = 0
    diff_f0_loss = 0
    vae_loss = 0
    
    with torch.no_grad():
        for batch_idx, (y, norm_y_f0, y_f0, gender) in enumerate(eval_loader):
            y = y.cuda(0)
            norm_y_f0 = norm_y_f0.cuda(0)
            y_f0 = y_f0.cuda(0)
            gender = gender.squeeze().unsqueeze(0).cuda(0)
            

            mel_y = mel_fn(y)
            f0_y = torch.tensor(y_f0, dtype=torch.float32)
            length = torch.LongTensor([mel_y.size(2)]).cuda(0)

            expected_f0_len = mel_y.size(2) * 4
            current_f0_len = f0_y.size(2)

            if current_f0_len > expected_f0_len:
                f0_y = f0_y[:, :, :expected_f0_len]
                norm_y_f0 = norm_y_f0[:, :, :expected_f0_len]
            elif current_f0_len < expected_f0_len:
                pad_len = expected_f0_len - current_f0_len
                f0_y = F.pad(f0_y, (0, pad_len), "constant", 0)
                norm_y_f0 = F.pad(norm_y_f0, (0, pad_len), "constant", 0)

            y_pad = F.pad(y, (40, 40), "reflect")
            w2v_y = w2v(y_pad)

            y_f0_hat, y_mel, o_f0, o_mel, vae = model(mel_y, w2v_y, norm_y_f0, f0_y, length, gender, n_timesteps=6, mode='ml')

            mel_loss += F.l1_loss(mel_y, o_mel).item()
            enc_loss += F.l1_loss(mel_y, y_mel).item()
            enc_f0_loss += F.l1_loss(f0_y, y_f0_hat).item()
            diff_f0_loss += F.l1_loss(f0_y, o_f0).item()
            vae_loss += vae.item()

            generated = model.module.infer_vc(
                mel_y, 
                w2v_y, 
                norm_y_f0, 
                f0_y, 
                length, 
                diffpitch_ts=30, 
                diffvoice_ts=6, 
                fixed_noise=fixed_noise, 
                gender=gender)

            if batch_idx > 100:
                break
            if batch_idx <= 4:
                y_real = net_v(mel_y)
                y_hat = net_v(o_mel)
                enc_hat = net_v(y_mel)
                generated = net_v(generated)
                folder = './audio_gen/centralized'
                os.makedirs(folder, exist_ok=True)
                save_audio(y_real, f'{folder}/real_{batch_idx}.wav', syn_sr=hps.data.sampling_rate)
                save_audio(y_hat, f'{folder}/reconstr_{batch_idx}.wav', syn_sr=hps.data.sampling_rate)
                save_audio(enc_hat, f'{folder}/encoded_{batch_idx}.wav', syn_sr=hps.data.sampling_rate)
                save_audio(generated, f'{folder}/gen_{batch_idx}.wav', syn_sr=hps.data.sampling_rate)
                audio_dict.update({
                    "gen/audio_{}".format(batch_idx): y_hat.squeeze(),
                    "gen/enc_audio_{}".format(batch_idx): enc_hat.squeeze()
                })
                if global_step == 0:
                    audio_dict.update({"gt/audio_{}".format(batch_idx): y.squeeze()})

        mel_loss /= 100
        enc_loss /= 100
        enc_f0_loss /= 100
        diff_f0_loss /= 100
        vae_loss /= 100
        
    scalar_dict = {
        "val/mel": mel_loss, 
        "val/enc_mel": enc_loss, 
        "val/enc_f0": enc_f0_loss, 
        "val/diff_f0": diff_f0_loss, 
        "val/vae": vae_loss}
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
        scalars=scalar_dict
    )
    model.train()
    return mel_loss, enc_loss, enc_f0_loss, diff_f0_loss, vae_loss

def get_hparams_from_dict(config_dict):
    hps = types.SimpleNamespace()
    for section, params in config_dict.items():
        setattr(hps, section, types.SimpleNamespace(**params))
    hps.model_dir = "./checkpoints/centralized"
    return hps

def get_hparams():
    parser = argparse.ArgumentParser(description="DIVAS training configuration")

    parser.add_argument('--config', type=str, required=True, help="Path to config JSON file")
    parser.add_argument('--vocoder', type=str, default='hifigan', choices=['hifigan', 'bigvgan'], help="Choose vocoder")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    config_dict["vocoder"]["voc"] = args.vocoder
    config_dict["vocoder"]["ckpt_voc"] = (
        "./vocoder/voc_hifigan.pth" if args.vocoder == "hifigan" else "./vocoder/voc_bigvgan.pth"
    )

    return get_hparams_from_dict(config_dict)

def save_audio(wav, out_file, syn_sr=16000):
    wav = (wav.squeeze() / wav.abs().max() * 0.999 * 32767.0).cpu().numpy().astype('int16')
    write(out_file, syn_sr, wav) 

# %% 
if __name__ == "__main__":
    hps = get_hparams()
    print("Preparing training...")
    main(hps)

# %%
