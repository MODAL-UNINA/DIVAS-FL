# %%
import os
import argparse
import types
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import random
import json

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import flwr as fl

from utils.data_loader import AudioDataset, MelSpectrogramFixed
from utils import utils
from model.DIVAS import Wav2vec2, DIVAS
from vocoder.hifigan import HiFi
from vocoder.bigvgan import BigvGAN
from augmentation.aug import Augment

import logging
import warnings
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("flwr").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

# -----------------------------
# PARSER ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser(description="Start Flower server.")
parser.add_argument('--num_rounds', type=int, default=10, help='Number of federated learning rounds')
parser.add_argument('--min_available_clients', type=int, default=5, help='Minimum number of clients available')
parser.add_argument('--min_fit_clients', type=int, default=5, help='Minimum number of clients to train each round')
parser.add_argument('--device', type=str, default='cuda:0', help='GPU device ID to use (default: 0)')
parser.add_argument('--server_address', type=str, default='localhost:39705', help='Server address for Flower')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for (server) evaluation')
parser.add_argument('--vocoder', type=str, default='hifigan', choices=['hifigan', 'bigvgan'], help='Vocoder type to use')
parser.add_argument('--data_path', type=str, default='./run_txt/server', help='Path to the dataset directory')
args = parser.parse_args()

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def init_model_for_server(hps, device):
    logger = utils.get_logger(hps.model_dir)
    model, _, _, _, _, _ = init_model_components(hps, logger, device)
    return model

def init_model_components(hps, logger, device):
    """Initialize model components."""
    # Mel spectrogram extractor
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).to(device)
    
    # Wav2vec2 encoder
    w2v = Wav2vec2().to(device)

    # Augmentation module
    aug = Augment(hps).to(device)
    
    # Main model
    model = DIVAS(
        hps.data.n_mel_channels,
        hps.diffusion.spk_dim,
        hps.diffusion.dec_dim,
        hps.diffusion.beta_min,
        hps.diffusion.beta_max,
        hps
    ).to(device)
    
    # Vocoder (for evaluation)
    if hps.vocoder.voc == "bigvgan":
        net_v = BigvGAN(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **vars(hps.model)
        ).to(device)
    else:
        net_v = HiFi(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **vars(hps.model)
        ).to(device)
    
    # Load vocoder checkpoint
    try:
        utils.load_checkpoint(hps.vocoder.ckpt_voc, net_v, None)
        net_v.eval()
        net_v.dec.remove_weight_norm()
    except Exception as e:
        logger.warning(f"Could not load vocoder checkpoint: {e}")

    # Gradient scaler for mixed precision
    scaler = GradScaler('cuda', enabled=hps.train.fp16_run)

    logger.info(f"Model initialized with {get_param_num(model)} parameters")
    
    pretrained_path = './models/pretrained_server.pth'
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        utils.load_checkpoint(pretrained_path, model, None)
    else:
        logger.warning(f"Pretrained model not found at {pretrained_path}. Normal Federated Learning.")

    return model, w2v, mel_fn, net_v, aug, scaler

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def get_hparams_from_dict(config_dict):
    """Convert config dict to hyperparameters namespace."""
    hps = types.SimpleNamespace()
    for section, params in config_dict.items():
        setattr(hps, section, types.SimpleNamespace(**params))
    hps.model_dir = "./checkpoints/server"
    os.makedirs(hps.model_dir, exist_ok=True)
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

def build_config(data_path, vocoder, args, overrides=None):
    """Build a dynamic CONFIG dict based on client ID and data path."""
    if vocoder == 'hifigan':
        upsample_rates = [5, 4, 4, 2, 2]
        upsample_initial_channel = 512
        upsample_kernel_sizes = [11, 8, 8, 4, 4]
        ckpt_voc = "./vocoder/voc_hifigan.pth"
    elif vocoder == 'bigvgan':
        upsample_rates = [5, 4, 2, 2, 2, 2]
        upsample_initial_channel = 1024
        upsample_kernel_sizes = [11, 8, 4, 4, 4, 4]
        ckpt_voc = "./vocoder/voc_bigvgan.pth"
    else:
        raise ValueError(f"Unsupported vocoder: {vocoder}")
    
    config = {
        "train": {
            "eval_interval": 10,
            "seed": 1234,
            "epochs": 1000,
            "optimizer": "adamw",
            "lr_decay_on": True,
            "learning_rate": 5e-5,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": args.batch_size,
            "fp16_run": False,
            "lr_decay": 0.999875,
            "segment_size": 35840,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 1,
            "aug": True,
            "lambda_commit": 0.02
        },
        "data": {
            "train_filelist_path": os.path.join(data_path, "train_wav.txt"),
            "test_filelist_path": os.path.join(data_path, "test_wav.txt"),
            "sampling_rate": 16000,
            "filter_length": 1280,
            "hop_length": 320,
            "win_length": 1280,
            "n_mel_channels": 80,
            "mel_fmin": 0,
            "mel_fmax": 8000
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "mixup_ratio": 0.6,
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "hidden_size": 128,
            "latent_dim": 50,
        },
        "diffusion": {
            "dec_dim": 64,
            "spk_dim": 128,
            "beta_min": 0.05,
            "beta_max": 20.0
        },
        "vocoder": {
            "voc": "hifigan",
            "ckpt_voc": ckpt_voc
        },
    }
    
    # Apply overrides if provided
    if overrides:
        for section, values in overrides.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
    
    return config

def get_evaluate_fn(hps, device):
    
    
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    
    model, w2v, mel_fn, _, _, _ = init_model_components(hps, logger, device)
    print("Server Model initialized")

    # Carica il validation loader (server side)
    test_loader, logger = load_server_data(hps, logger)

    history = {
        "test_mel_diff": [], 
        "test_f0_diff": [], 
        "test_vae": [], 
    }

    def evaluate(server_round, parameters, config):
        nonlocal test_loader

        params_ndarrays = parameters.tensors if hasattr(parameters, "tensors") else parameters
        params_dict = zip(model.state_dict().keys(), params_ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Evaluate on test set
        loss_test = evaluate_model(model, mel_fn, w2v, test_loader, device, logger)
        mel_diff = loss_test["mel_diff"]
        f0_diff = loss_test["f0_diff"]
        vae = loss_test["vae"]
        print(f"[Server] Round {server_round} -> Test Loss: mel_diff={mel_diff:.4f}, f0_diff={f0_diff:.4f}, vae={vae:.4f}")

        # Store metrics in history
        history["test_mel_diff"].append(mel_diff)
        history["test_f0_diff"].append(f0_diff)
        history["test_vae"].append(vae)

        plot_losses(history, dataset="test", hps=hps)

        # save model checkpoint
        model_path = os.path.join(hps.model_dir, f"model_round_{server_round}.pth")
        torch.save(model.state_dict(), model_path)

        logger.info(f"Server - Epoch losses: {loss_test}")
        # Return loss and metrics for Flower
        return loss_test, {
            "mel_diff": mel_diff,
            "f0_diff": f0_diff,
            "vae": vae
        }

    return evaluate

def plot_losses(history, dataset, hps):
    """Plot training and validation curves over all epochs."""
    # Crea un numero di plot in base alla self.history
    if not history[f"{dataset}_mel_diff"]:
        print(f"Server: No {dataset} history available to plot.")
        return

    num_plots = 3
    plt.figure(figsize=(15, 5 * num_plots))
    plt.subplots_adjust(hspace=0.4)
    for i, (key, values) in enumerate(history.items()):
        if key.startswith(f"{dataset}_") and values:
            plt.subplot(num_plots, 1, i + 1)
            plt.plot(values, label=key.replace(f"{dataset}_", "").replace("_", " ").title())
            plt.title(f"{dataset.capitalize()} {key.replace(f'{dataset}_', '').replace('_', ' ').title()}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
    plt.savefig(os.path.join(hps.model_dir, f"{dataset}_training_curves.png"))
    plt.close()

def evaluate_model(model, mel_fn, w2v, loader, device, logger):
    """Evaluate the model on the given DataLoader."""
    model.eval()
    losses = {"mel_diff": 0.0, "f0_diff": 0.0, "vae": 0.0}
    max_batches = len(loader)
    
    batches_to_evaluate = set(random.sample(range(max_batches), min(100, max_batches)))
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (y, norm_y_f0, y_f0, gender) in enumerate(loader):
            if batch_idx not in batches_to_evaluate:  # Limit evaluation samples
                continue
            try:
                y = y.to(device, non_blocking=True)
                norm_y_f0 = norm_y_f0.to(device, non_blocking=True)
                y_f0 = y_f0.to(device, non_blocking=True)
                gender = gender.squeeze().unsqueeze(0).to(device, non_blocking=True)
                
                mel_y = mel_fn(y)
                f0_y = torch.tensor(y_f0, dtype=torch.float32).to(device)
                length = torch.LongTensor([mel_y.size(2)]).to(device)
                
                # Adjust F0 length to match mel
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
                
                # Forward pass
                _, _, o_f0, o_mel, vae_loss = model(
                    mel_y, w2v_y, norm_y_f0, f0_y, length, gender,
                    n_timesteps=6, mode='ml'
                )
                
                # Compute losses
                mel_loss = F.l1_loss(mel_y, o_mel)
                f0_loss = F.l1_loss(f0_y, o_f0)
                
                losses["mel_diff"] += mel_loss.item()
                losses["f0_diff"] += f0_loss.item()
                losses["vae"] += vae_loss.item()
                num_samples += 1
            except Exception as e:
                logger.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue

    for key in losses:
        losses[key] /= len(batches_to_evaluate)

    return losses

def save_audio(wav, out_file, syn_sr=16000):
    wav = (wav.squeeze() / wav.abs().max() * 0.999 * 32767.0).cpu().numpy().astype('int16')
    write(out_file, syn_sr, wav) 

# -----------------------------
# LOAD SERVER DATA
# -----------------------------
def load_server_data(hps, logger):
    """Initialize data loader for the server."""

    try:
        # Test dataset
        num_workers = min(4, os.cpu_count() // 2)
        test_dataset = AudioDataset(hps, training=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        logger.info(f"Test data loader initialized with {len(test_dataset)} test samples")
    except Exception as e:
        logger.error(f"Failed to initialize data loader: {e}")
        raise
    return test_loader, logger

# -----------------------------
# SERVER SETUP AND START
# -----------------------------
def setup_server_data(args):
    import re
    """Aggregate test files from all clients into unified server copies."""
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    root_dir = os.path.dirname(os.path.abspath(args.data_path))

    # File names we're aggregating
    filenames = ["test_f0_norm.txt", "test_f0.txt", "test_wav.txt"]
    contents = {name: [] for name in filenames}

    # Stats aggregation
    total_speakers = 0
    male_speakers = 0
    male_files = 0
    female_speakers = 0
    female_files = 0
    total_seconds = 0.0

    for client_dir in os.listdir(root_dir):
        if client_dir.startswith("client_"):
            client_path = os.path.join(root_dir, client_dir)
            if not os.path.isdir(client_path):
                continue

            # Aggrega test file (concatena contenuti)
            for fname in filenames:
                client_file = os.path.join(client_path, fname)
                if os.path.exists(client_file):
                    with open(client_file, "r", encoding="utf-8") as f:
                        contents[fname].append(f.read())

            # Aggrega test_stats numericamente
            stats_file = os.path.join(client_path, "test_stats.txt")
            if os.path.exists(stats_file):
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats = f.read()

                    # Estrai numeri con regex
                    total_speakers += int(re.search(r"Total speakers: (\d+)", stats).group(1))
                    male_match = re.search(r"Male: (\d+) speaker, (\d+) file", stats)
                    female_match = re.search(r"Female: (\d+) speaker, (\d+) file", stats)
                    duration_match = re.search(r"Durata totale: [\d.]+ ore \(([\d.]+) s\)", stats)

                    if male_match:
                        male_speakers += int(male_match.group(1))
                        male_files += int(male_match.group(2))
                    if female_match:
                        female_speakers += int(female_match.group(1))
                        female_files += int(female_match.group(2))
                    if duration_match:
                        total_seconds += float(duration_match.group(1))

    # Scrivi i file aggregati nel server
    for fname, data in contents.items():
        output_file = os.path.join(args.data_path, fname)
        # Rimuove righe vuote e spazi finali
        normalized = [line.strip() for content in data for line in content.strip().splitlines() if line.strip()]
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(normalized))

    # Calcola durata in ore
    total_hours = total_seconds / 3600 if total_seconds else 0.0
    perc_female = 100 * female_files / (female_files + male_files) if (female_files + male_files) > 0 else 0

    # Scrivi file stats aggregato
    stats_output = os.path.join(args.data_path, "test_stats.txt")
    with open(stats_output, "w", encoding="utf-8") as f:
        f.write(f"Total speakers: {total_speakers}\n")
        f.write(f"Male: {male_speakers} speaker, {male_files} file\n")
        f.write(f"Female: {female_speakers} speaker, {female_files} file ({perc_female:.1f}%)\n")
        f.write(f"Durata totale: {total_hours:.2f} ore ({total_seconds:.2f} s)\n")

    print(f"✅ Aggregated statistics saved in: {stats_output}")


# %%
# -----------------------------
# SERVER START
# -----------------------------
if __name__ == "__main__":
    hps = build_config(args.data_path, args.vocoder, args)
    hps = get_hparams_from_dict(hps)
    setup_server_data(args)
    
    # Carica il modello preallenato
    model = init_model_for_server(hps, args.device)

    # Estrai i pesi in formato Flower
    initial_parameters = fl.common.ndarrays_to_parameters(
        [param.detach().cpu().numpy() for param in model.state_dict().values()]
    )

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=args.min_available_clients,
        min_fit_clients=args.min_fit_clients,
        evaluate_fn=get_evaluate_fn(hps, args.device),
        initial_parameters=initial_parameters
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
