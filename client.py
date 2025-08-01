import os
import types
import logging
import warnings
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import flwr as fl

from utils.data_loader import AudioDataset, MelSpectrogramFixed
from utils import utils
from model.DIVAS import Wav2vec2, DIVAS
from vocoder.hifigan import HiFi
from vocoder.bigvgan import BigvGAN
from augmentation.aug import Augment
import module.commons as commons

import time
import random


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("flwr").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

class DIVASClient(fl.client.NumPyClient):
    """Client class for the DIVAS federated learning framework."""
    
    def __init__(self, client_id, data_path, args):
        """
        Initialize the DIVAS client.
        
        Args:
            client_id (str): Unique identifier for the client.
            data_path (str): Path to the client's data.
            device (str): Device to run the computations on (default: "cpu").
        """
        self.t0 = time.time()
        self.client_id = client_id
        self.data_path = os.path.join(data_path, client_id)
        self.args = args
        self.device = self.args.device
        print(f"[{self.client_id}] Initializing client...")
        
        # Build configuration
        CONFIG = build_config(self.data_path, vocoder='hifigan', args=args)
        self.hps = get_hparams_from_dict(CONFIG, self.client_id)
        print(f"[{self.client_id}] Model config loaded")
        
        # Setup logging and tensorboard
        self.logger = utils.get_logger(self.hps.model_dir)
        self.logger.info(self.hps)
        utils.check_git_hash(self.hps.model_dir)
        self.writer = SummaryWriter(log_dir=self.hps.model_dir)
        self.writer_eval = SummaryWriter(log_dir=os.path.join(self.hps.model_dir, "eval"))

        # Initialize model components
        self._init_model()
        print(f"[{self.client_id}] Model initialized")
        self._init_data_loader()
        print(f"[{self.client_id}] Data loader initialized")
        self._init_optimizer()
        print(f"[{self.client_id}] Optimizer initialized")

        self.t1 = time.time()
        print(f"[{self.client_id}] Initialization completed in {self.t1 - self.t0:.2f} seconds")

        # Training state
        self.round_counter = 0
        self.global_step = 0
        self._param_cache = None


    def _init_model(self):
        """Initialize model components."""
        # Mel spectrogram extractor
        self.mel_fn = MelSpectrogramFixed(
            sample_rate=self.hps.data.sampling_rate,
            n_fft=self.hps.data.filter_length,
            win_length=self.hps.data.win_length,
            hop_length=self.hps.data.hop_length,
            f_min=self.hps.data.mel_fmin,
            f_max=self.hps.data.mel_fmax,
            n_mels=self.hps.data.n_mel_channels,
            window_fn=torch.hann_window
        ).to(self.device)
        
        # Wav2vec2 encoder
        self.w2v = Wav2vec2().to(self.device)

        # Augmentation module
        self.aug = Augment(self.hps).to(self.device)
        
        # Main model
        self.model = DIVAS(
            self.hps.data.n_mel_channels,
            self.hps.diffusion.spk_dim,
            self.hps.diffusion.dec_dim,
            self.hps.diffusion.beta_min,
            self.hps.diffusion.beta_max,
            self.hps
        ).to(self.device)
        
        # Vocoder (for evaluation)
        if self.hps.vocoder.voc == "bigvgan":
            self.net_v = BigvGAN(
                self.hps.data.n_mel_channels,
                self.hps.train.segment_size // self.hps.data.hop_length,
                **vars(self.hps.model)
            ).to(self.device)
        else:
            self.net_v = HiFi(
                self.hps.data.n_mel_channels,
                self.hps.train.segment_size // self.hps.data.hop_length,
                **vars(self.hps.model)
            ).to(self.device)
        
        # Load vocoder checkpoint
        try:
            utils.load_checkpoint(self.hps.vocoder.ckpt_voc, self.net_v, None)
            self.net_v.eval()
            self.net_v.dec.remove_weight_norm()
        except Exception as e:
            self.logger.warning(f"Could not load vocoder checkpoint: {e}")

        # Gradient scaler for mixed precision
        self.scaler = GradScaler('cuda', enabled=self.hps.train.fp16_run)

        self.logger.info(f"Model initialized with {get_param_num(self.model)} parameters")

    def _init_data_loader(self):
        """Initialize data loader for this client."""
        try:
            # Increase num_workers for better I/O performance
            num_workers = min(4, os.cpu_count() // 2)

            # Create client-specific dataset
            self.train_dataset = AudioDataset(self.hps, training=True)
            self.train_loader = CustomDataLoader(
                self.train_dataset,
                batch_size=self.hps.train.batch_size,
                shuffle=True,
                drop_last=True,
                type='train'
            )
            
            # Test dataset (optional)
            self.test_dataset = AudioDataset(self.hps, training=False)
            self.test_loader = CustomDataLoader(
                self.test_dataset,
                batch_size=1,  # Test batch size is usually 1
                shuffle=False,
                type='test'
            )
            
            self.logger.info(f"Train data loader initialized with {len(self.train_dataset)} training samples")
            self.logger.info(f"Test data loader initialized with {len(self.test_dataset)} test samples")
        except Exception as e:
            self.logger.error(f"Failed to initialize data loader: {e}")
            raise
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hps.train.learning_rate,
            betas=self.hps.train.betas,
            eps=self.hps.train.eps
        )
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.hps.train.lr_decay
        )

    def get_parameters(self, config):
        """Optimized parameter extraction."""
        if self._param_cache is None:
            self._param_cache = [val.cpu().numpy() for val in self.model.state_dict().values()]
        else:
            for i, val in enumerate(self.model.state_dict().values()):
                self._param_cache[i] = val.cpu().numpy()
        return self._param_cache
    
    def set_parameters(self, parameters):
        """Optimized parameter setting."""
        state_dict = self.model.state_dict()
        for k, val in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(val, device=self.device)
        self.model.load_state_dict(state_dict)
        # Invalidate cache
        self._param_cache = None
    
    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        # Set received parameters
        self.set_parameters(parameters)
        self.round_counter += 1
        self.local_epochs = self.args.epochs # Example: number of local epochs per round
        print(f"[Client {self.client_id}] Starting training round {self.round_counter}")

        # Training loop
        self.model.train()
        losses = self._train_epoch(self.local_epochs)

        # Plot training curves
        self._plot_training_curves(losses, subset="train")

        # TODO: SALVARE LA STORIA ?? Non serve visto il self?
        return self.get_parameters(config), len(self.train_loader), {
            "train_mel_diff": float(losses["mel_diff"]),
            "train_f0_diff": float(losses["f0_diff"]),
            "train_vae": float(losses["vae"]),
            "round": self.round_counter
        }

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test dataset."""
        self.set_parameters(parameters)
        losses = self.evaluate_model(self.test_loader)
        self._plot_training_curves(losses, subset="test")
        print(f"[Client {self.client_id}] Evaluation complete")
        return losses["mel_diff"], len(self.test_loader), {
            "mel_diff": float(losses["mel_diff"]),
            "f0_diff": float(losses["f0_diff"]),
            "vae": float(losses["vae"]),
            "round": self.round_counter
        }
    
    def _plot_training_curves(self, losses, subset):
        """Plot training and validation curves over all epochs."""
        
        if not hasattr(self, '_loss_history_train'):
            self._loss_history_train = {key: [] for key in losses.keys()}
        if not hasattr(self, '_loss_history_test'):
            self._loss_history_test = {key: [] for key in losses.keys()}

        loss_dict = self._loss_history_train if subset == "train" else self._loss_history_test

        for key, value in losses.items():
            loss_dict[key].append(value)
        
        plt.figure(figsize=(15, 5))
        for i, (key, values) in enumerate(loss_dict.items()):
            plt.subplot(1, 3, i + 1)
            plt.plot(values, label=key.replace("_", " ").title())
            plt.title(f"Training {key.replace('_', ' ').title()}")
            plt.xlabel("Rounds")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.hps.model_dir, f"{subset}_training_curves.png"))
        plt.close()
    
    def _train_epoch(self, local_epochs):
        """Train for specified number of local epochs."""
        losses = {"mel_diff": 0.0, "f0_diff": 0.0, "vae": 0.0}
        num_batches = 0
        
        for epoch in range(local_epochs):
            self.logger.info('====> Epoch: {}'.format(epoch))
            t0 = time.time()
            for batch_idx, (x, norm_f0, x_f0, length, gender) in enumerate(self.train_loader):
                try:
                    x = x.to(self.device, non_blocking=True)
                    norm_f0 = norm_f0.to(self.device, non_blocking=True)
                    x_f0 = x_f0.to(self.device, non_blocking=True)
                    length = length.to(self.device, non_blocking=True).squeeze()
                    gender = gender.to(self.device, non_blocking=True).squeeze()
                    
                    
                    # Process audio
                    mel_x = self.mel_fn(x)
                    aug_x = self.aug(x)
                    x = x if torch.isnan(aug_x).any() else aug_x
                    x_pad = F.pad(x, (40, 40), "reflect")
                    w2v_x = self.w2v(x_pad)
                    f0_x = torch.tensor(x_f0, dtype=torch.float32).to(self.device)
                
                    # Forward pass and compute loss
                    self.optimizer.zero_grad()
                
                    loss_mel_diff, loss_mel_diff_rec, loss_f0_diff, loss_mel, loss_f0, loss_vae = \
                        self.model.compute_loss(mel_x, w2v_x, norm_f0, f0_x, length, gender)
                
                    loss_total = (loss_mel_diff + loss_mel_diff_rec + loss_f0_diff + 
                        loss_mel * self.hps.train.c_mel + loss_f0 + loss_vae)
                    
                    # Backward pass
                    if self.hps.train.fp16_run:
                        self.scaler.scale(loss_total).backward()
                        self.scaler.unscale_(self.optimizer)
                        commons.clip_grad_value_(self.model.parameters(), None)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss_total.backward()
                        commons.clip_grad_value_(self.model.parameters(), None)
                        self.optimizer.step()
                    
                    # Accumulate losses
                    losses["mel_diff"] += loss_mel_diff.item()
                    losses["f0_diff"] += loss_f0_diff.item()
                    losses["vae"] += loss_vae.item()
                    num_batches += 1
                    
                    self.global_step += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error in training batch {batch_idx}: {e}")
                    continue
                
            t1 = time.time()
            print(f"[Client {self.client_id} Epoch completed in {t1 - t0:.2f} seconds")
            
            
            # Average losses
            if num_batches > 0:
                for key in losses:
                    losses[key] /= num_batches   

            if epoch % 5 == 0 or epoch == local_epochs - 1:
                utils.save_checkpoint(
                    self.model, self.optimizer, self.hps.train.learning_rate, epoch,
                    os.path.join(self.hps.model_dir, "G_actual.pth")
                )    
            self.logger.info(f"Client {self.client_id} - Epoch {epoch} losses: {losses}") 
            
            # Step scheduler
            self.scheduler.step()


        self.logger.info(f"Client {self.client_id} - Round {self.round_counter} Epoch losses: {losses}")
        utils.save_checkpoint(self.model, self.optimizer, self.hps.train.learning_rate, self.round_counter,
        os.path.join(self.hps.model_dir, f"G_round{self.round_counter}.pth"))
        return losses

    def evaluate_model(self, loader):
        """Evaluate the model on the given DataLoader."""
        self.model.eval()
        losses = {"mel_diff": 0.0, "f0_diff": 0.0, "vae": 0.0}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (y, norm_y_f0, y_f0, gender) in enumerate(loader):
                if batch_idx >= 20:  # Limit evaluation samples
                    break
                
                try:
                    y = y.to(self.device)
                    norm_y_f0 = norm_y_f0.to(self.device)
                    y_f0 = y_f0.to(self.device)
                    gender = gender.squeeze().unsqueeze(0).to(self.device)
                    
                    mel_y = self.mel_fn(y)
                    f0_y = torch.tensor(y_f0, dtype=torch.float32).to(self.device)
                    length = torch.LongTensor([mel_y.size(2)]).to(self.device)
                    
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
                    w2v_y = self.w2v(y_pad)
                    
                    # Forward pass
                    y_f0_hat, y_mel, o_f0, o_mel, vae_loss = self.model(
                        mel_y, w2v_y, norm_y_f0, f0_y, length, gender,
                        n_timesteps=6, mode='ml'
                    )
                    
                    # Compute losses
                    mel_loss = F.l1_loss(mel_y, o_mel)
                    f0_loss = F.l1_loss(f0_y, o_f0)
                    
                    losses["mel_diff"] += mel_loss.item()
                    losses["f0_diff"] += f0_loss.item()
                    losses["vae"] += vae_loss.item()
                    num_batches += 1
                    if batch_idx <= 4:
                        fixed_noise = torch.randn(1, self.model.latent_dim, device=self.device)
                        # generated = self.model.module.infer_vc(
                        generated = self.model.infer_vc(
                        mel_y, 
                        w2v_y, 
                        norm_y_f0, 
                        f0_y, 
                        length, 
                        diffpitch_ts=30, 
                        diffvoice_ts=4, 
                        fixed_noise=fixed_noise, 
                        diversity_scale=1.0,
                        gender=gender)
                        y_real = self.net_v(mel_y)
                        y_hat = self.net_v(o_mel)
                        enc_hat = self.net_v(y_mel)
                        generated = self.net_v(generated)
                        folder = f'./audio_gen/{self.client_id}'
                        os.makedirs(folder, exist_ok=True)
                        save_audio(y_real, f'{folder}/real_{batch_idx}.wav', syn_sr=self.hps.data.sampling_rate)
                        save_audio(y_hat, f'{folder}/reconstr_{batch_idx}.wav', syn_sr=self.hps.data.sampling_rate)
                        save_audio(enc_hat, f'{folder}/encoded_{batch_idx}.wav', syn_sr=self.hps.data.sampling_rate)
                        save_audio(generated, f'{folder}/gen_{batch_idx}.wav', syn_sr=self.hps.data.sampling_rate)

                except Exception as e:
                    self.logger.warning(f"Error in evaluation batch {batch_idx}: {e}")
                    continue

        if num_batches > 0:
            for key in losses:
                losses[key] /= num_batches

        self.logger.info(f"Client {self.client_id} - Test epoch losses: {losses}")
        return losses


class CustomDataLoader:
    def __init__(self,
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                collate_fn=None,
                type='train'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else self.my_collate_fn
        self.type = type
        self.indices = list(range(len(self.dataset)))

    def __len__(self):
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idxs = self.indices.copy()
        if self.shuffle:
            random.shuffle(idxs)

        batch = []
        for idx in idxs:
            batch.append(idx)
            if len(batch) == self.batch_size:
                samples = [self.dataset[i] for i in batch]
                yield self.collate_fn(samples, self.type)
                batch = []
        if (not self.drop_last) and len(batch) > 0:
            samples = [self.dataset[i] for i in batch]
            yield self.collate_fn(samples, self.type)
    
    @staticmethod
    def my_collate_fn(batch, type):
        if type == 'train':
            x1, x2, x3, x4, x5 = zip(*batch)
            return (
                torch.stack(x1),
                torch.stack(x2),
                torch.stack(x3),
                torch.stack(x4),
                torch.stack(x5),
            )
        elif type == 'test':
            y1, y2, y3, y4 = zip(*batch)
            return (
                torch.stack(y1),
                torch.stack(y2),
                torch.stack(y3),
                torch.stack(y4),
            )

def save_audio(wav, out_file, syn_sr=16000):
    wav = (wav.squeeze() / wav.abs().max() * 0.999 * 32767.0).cpu().numpy().astype('int16')
    write(out_file, syn_sr, wav) 

def get_hparams_from_dict(config_dict, client_id):
    """Convert config dict to hyperparameters namespace."""
    hps = types.SimpleNamespace()
    for section, params in config_dict.items():
        setattr(hps, section, types.SimpleNamespace(**params))
    hps.model_dir = f"./checkpoints/{client_id}"
    os.makedirs(hps.model_dir, exist_ok=True)
    return hps

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

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