import os
import torch
from torch.nn import functional as F

from scipy.io.wavfile import write
import utils
import json
import argparse

from model.DIVAS import Wav2vec2, DIVAS
from utils.data_loader import MelSpectrogramFixed
from utils import utils
from vocoder.hifigan import HiFi
from vocoder.bigvgan import BigvGAN
import types
import torchaudio
import numpy as np
import pandas as pd

import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic 

import logging
import warnings

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Evaluation libraries
from msclap import CLAP
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve
import whisper
from jiwer import wer


logging.basicConfig(level=logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


torch.backends.cudnn.benchmark = True
global_step = 0


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def get_hparams_from_dict2(config_dict):
    return types.SimpleNamespace(**config_dict)

def get_hparams():
    parser = argparse.ArgumentParser(description="DIVAS training configuration")

    parser.add_argument('--config', type=str, required=True, help="Path to config JSON file")
    parser.add_argument('--vocoder', type=str, default='hifigan', choices=['hifigan', 'bigvgan'], help="Choose vocoder")
    parser.add_argument('--target', type=str, default='server', choices=['server', 'pretrained_server'], help="Target for inference (e.g., 'server')")
    parser.add_argument('--flag', action='store_true', help="Run generation flag")
    parser.add_argument('--number', type=int, default=1, help="Number of samples to generate for each audio file")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    config_dict["vocoder"]["voc"] = args.vocoder
    config_dict["vocoder"]["ckpt_voc"] = (
        "./vocoder/voc_hifigan.pth" if args.vocoder == "hifigan" else "./vocoder/voc_bigvgan.pth"
    )

    return get_hparams_from_dict(config_dict), args

def get_yaapt_f0(audio, sr=16000, interp=False):
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0) 
        pitch = pYAAPT.yaapt(basic.SignalObj(y_pad, sr), 
                             **{'frame_length': 20.0, 'frame_space': 5.0, 'nccf_thresh1': 0.25, 'tda_frame_length': 25.0})
        f0s.append(pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :])

    return np.vstack(f0s)  


def get_hparams_from_dict(config_dict):
    hps = types.SimpleNamespace()
    for section, params in config_dict.items():
        setattr(hps, section, types.SimpleNamespace(**params))
    hps.model_dir = "./checkpoints"
    return hps


def save_audio(wav, out_file, syn_sr=16000):
    if hasattr(wav, 'abs'):  # PyTorch tensor
        max_val = wav.abs().max()
        wav = (wav.squeeze() / max_val * 0.999 * 32767.0).cpu().numpy()
    else:  # NumPy array
        wav = np.squeeze(wav)
        max_val = np.abs(wav).max()
        wav = (wav / max_val * 0.999 * 32767.0)
    wav = wav.astype('int16')
    write(out_file, syn_sr, wav)


def load_audio(path):
    audio, sr = torchaudio.load(path) 
    audio = audio[:1]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000, resampling_method="kaiser_window")
    
    p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1] 
    audio = torch.nn.functional.pad(audio, (0, p)) 

    return audio 


def generate_audio_target(path, gender, mel_fn, w2v, net_v, model, gender_name, hps, config, number=1, save_real=True, COUNT=0):
    """Generate audio from a given path."""
    
    speaker_id = os.path.normpath(path).split(os.sep)[-3]
    src_name = os.path.splitext(os.path.basename(path))[0]
    audio = load_audio(path)   

    src_mel = mel_fn(audio.cuda())
    real_voice = net_v(src_mel.cuda())
    
    src_length = torch.LongTensor([src_mel.size(-1)]).cuda()
    w2v_x = w2v(F.pad(audio, (40, 40), "reflect").cuda())

    try:
        f0 = get_yaapt_f0(audio.numpy())
    except:
        f0 = np.zeros((1, audio.shape[-1] // 80), dtype=np.float32)

    f0_x = f0.copy()
    f0_x = torch.log(torch.FloatTensor(f0_x+1)).cuda()
    ii = f0 != 0
    f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()
    f0_norm_x = torch.FloatTensor(f0).cuda()
    
    list_id = []
    list = []

    gender = torch.tensor(gender).unsqueeze(0).cuda()

    with torch.no_grad():
        for i in range(number):  # Genera "number" campioni
            noise = torch.randn(1, hps.model.latent_dim).cuda()
            COUNT += 1
            print(f'>> Generating sample {COUNT} ...')
            outputs = model.infer_vc(src_mel, 
                                    w2v_x, 
                                    f0_norm_x, 
                                    f0_x, 
                                    src_length, 
                                    diffpitch_ts=30, 
                                    diffvoice_ts=6, 
                                    fixed_noise=noise,
                                    diversity_scale=1.1, 
                                    gender=gender)
            converted_audio = net_v(outputs).detach().cpu().squeeze().numpy()
            converted_audio = converted_audio.reshape(-1, 1)

            list.append(converted_audio)
            list_id.append(converted_audio)
            
            # Salva i campioni generati
            f_name = f'{speaker_id}_{src_name}_generated_{i+1}_{gender_name}.wav'
            save_audio(converted_audio, os.path.join(config.output_dir, f_name))

        if len(list_id) > 1:
            for i in range(len(list_id) - 1):
                for j in range(i + 1, len(list_id)):
                    if torch.equal(list_id[i], list_id[j]):
                        print(f"Sample {i+1} and Sample {j+1} are equal.")
    
    if save_real:
        f_real = f'{speaker_id}_{src_name}_real.wav'
        save_real_path = os.path.join(os.path.dirname(config.output_dir), 'real_audio')
        os.makedirs(save_real_path, exist_ok=True)
        save_audio(real_voice.detach(), os.path.join(save_real_path, f_real))
    
    return COUNT


def generate(config, hps, args): 
    os.makedirs(config.output_dir, exist_ok=True) 
    number = args.number
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda()

    # Load pre-trained w2v (XLS-R)
    w2v = Wav2vec2().cuda()  
    
    # Load model
    model = DIVAS(hps.data.n_mel_channels, 
                hps.diffusion.spk_dim,
                hps.diffusion.dec_dim, 
                hps.diffusion.beta_min, 
                hps.diffusion.beta_max, 
                hps).cuda()

    model.load_state_dict(torch.load(config.ckpt_model)) # server post fine-tuning

    model.eval()
    
    # Load vocoder
    if hps.vocoder.voc == "hifigan":
        net_v = HiFi(hps.data.n_mel_channels, hps.train.segment_size // hps.data.hop_length, **vars(hps.model)).cuda()
        utils.load_checkpoint(hps.vocoder.ckpt_voc, net_v, None)
    elif hps.vocoder.voc == "bigvgan":
        net_v = BigvGAN(hps.data.n_mel_channels, hps.train.segment_size // hps.data.hop_length, **vars(hps.model)).cuda()
        utils.load_checkpoint(hps.vocoder.ckpt_voc, net_v, None)
    net_v.eval().dec.remove_weight_norm()  
    
    # TSNE visualization
    os.makedirs(config.output_dir, exist_ok=True)
    print(">> Visualizing sex space ...")
    visualize_sex_space(model.encoder, config.output_dir, n_samples = 400)

    # Generate audio 
    print('>> Generating ...')
    with open(config.src_path, 'r') as f:
        audio_paths = f.readlines()
    audio_paths = [path.strip() for path in audio_paths if path.strip()]
    config.output_dir = os.path.join(config.output_dir, 'generated_audio')
    os.makedirs(config.output_dir, exist_ok=True)
    COUNT = 0
    for audio_path in audio_paths:
        if not audio_path.endswith('.wav'):
            raise ValueError(f"Expected .wav file, got {audio_path}")
        audio_path = Path(audio_path).resolve()
        genders = [0, 1] # Male and Female
        for gender in genders:
            gender_name = 'male' if gender == 0 else 'female'
            COUNT = generate_audio_target(audio_path, 
                                gender, 
                                mel_fn, 
                                w2v, 
                                net_v, 
                                model, 
                                gender_name, 
                                hps=hps,
                                config=config,
                                number=number,
                                save_real=True,
                                COUNT=COUNT
                                )


def visualize_sex_space(model, folder, n_samples=400):

    genders = [0, 1]  # 0 = maschio, 1 = femmina
    noise_dim = model.style_vae.latent_dim

    all_points = []
    labels = []

    for gender_id in genders:
        # Prepara rumore
        noise = torch.randn(n_samples // 2, noise_dim).cuda()
        
        # Prepara condizione di genere
        gender_tensor = torch.tensor([gender_id] * (n_samples // 2)).long().cuda()
        gender_emb = model.gender_emb(gender_tensor)  # Shape: [N, cond_dim]
        
        # Flow su noise
        zK = model.style_vae.flow(noise)

        # Concatenazione con il condizionamento
        zK_cond = torch.cat((zK, gender_emb), dim=-1)  # Shape: [N, latent+cond]
        
        # Proiezione nello spazio stile finale
        # Use mixture of experts decoder
        expert_1_out = model.style_vae.expert_1(zK_cond)
        expert_2_out = model.style_vae.expert_2(zK_cond)
        gate = model.style_vae.gate_network(zK_cond)
        combined = gate[:, 0:1] * expert_1_out + gate[:, 1:2] * expert_2_out
        
        z_proj = model.style_vae.final_projection(combined).detach().cpu()
        
        all_points.append(z_proj)
        labels.extend([gender_id] * (n_samples // 2))

    zK_all = torch.cat(all_points, dim=0).numpy()
    labels = np.array(labels)
    cmap = plt.get_cmap('coolwarm')

    # TSNE per proiezione 2D
    tsne = TSNE(n_components=2, perplexity=30)
    z2d = tsne.fit_transform(zK_all)

    # Plot
    colors = [cmap(0.0), cmap(1.0)]
    labels_text = ['Male', 'Female']
    for gender_id, color, label_text in zip([0, 1], colors, labels_text):
        idx = labels == gender_id
        plt.scatter(z2d[idx, 0], z2d[idx, 1], c=color, label=label_text, alpha=0.7)

    # === TSNE Plot ===
    plt.figure(figsize=(8, 6))
    for gender_id, color, label_text in zip([0, 1], colors, labels_text):
        idx = labels == gender_id
        plt.scatter(z2d[idx, 0], z2d[idx, 1], c=color, label=label_text, alpha=0.7)

    plt.xlabel("TSNE 1", fontweight='bold')
    plt.ylabel("TSNE 2", fontweight='bold')
    plt.grid(True)
    patches = [mpatches.Patch(color=colors[i], label=labels_text[i]) for i in range(2)]
    plt.legend(handles=patches, prop={'weight': 'bold'})
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{folder}/sex_space_tsne.png', dpi=600)
    plt.close()

    # === PCA Plot ===
    pca = PCA(n_components=2)
    z2d_pca = pca.fit_transform(zK_all)

    plt.figure(figsize=(8, 6))
    for gender_id, color, label_text in zip([0, 1], colors, labels_text):
        idx = labels == gender_id
        plt.scatter(z2d_pca[idx, 0], z2d_pca[idx, 1], c=color, label=label_text, alpha=0.7)

    plt.xlabel("PCA 1", fontweight='bold')
    plt.ylabel("PCA 2", fontweight='bold')
    plt.grid(True)
    patches = [mpatches.Patch(color=colors[i], label=labels_text[i]) for i in range(2)]
    plt.legend(handles=patches, prop={'weight': 'bold'})
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{folder}/sex_space_pca.png', dpi=600)
    plt.close()

def get_real_name(generated_name):
    # Es: '00005_generated_1_female.wav' → '00005_real.wav'
    base = generated_name.split('_generated')[0]
    return f"{base}_real.wav"

def get_embeddings(folder, clap_model, batch_size=16):
    embeddings = {}
    all_files = [fname for fname in os.listdir(folder) if fname.endswith(".wav")]
    all_paths = [os.path.join(folder, fname) for fname in all_files]

    for i in tqdm(range(0, len(all_paths), batch_size), desc=f"Extracting from {folder}"):
        batch_paths = all_paths[i:i+batch_size]

        with torch.no_grad():
            embs = clap_model.get_audio_embeddings(batch_paths)

        for path, emb in zip(batch_paths, embs):
            fname = os.path.basename(path)
            embeddings[fname] = emb.cpu()
    return embeddings

def compute_clap_similarity(real_folder, generated_folder, batch_size=16):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clap = CLAP(version='2023', use_cuda=(device == 'cuda'))

    print("Extracting embeddings for real audio...")
    real_embs = get_embeddings(real_folder, clap, batch_size=batch_size)
    print("Extracting embeddings for generated audio...")
    gen_embs = get_embeddings(generated_folder, clap, batch_size=batch_size)

    scores = {
        'cosine': {},
        'l2': {},
        'l1': {}
    }

    for gen_name, gen_emb in gen_embs.items():
        real_name = get_real_name(gen_name)
        if real_name in real_embs:
            real_emb = real_embs[real_name]
            gen_emb = gen_emb

            # Ensure tensors are 2D
            real_emb = real_emb.unsqueeze(0)
            gen_emb = gen_emb.unsqueeze(0)

            cos = F.cosine_similarity(gen_emb, real_emb, dim=1).item()
            l2 = torch.norm(gen_emb - real_emb, p=2, dim=1).item()
            l1 = torch.norm(gen_emb - real_emb, p=1, dim=1).item()

            scores['cosine'][gen_name] = cos
            scores['l2'][gen_name] = l2
            scores['l1'][gen_name] = l1
        else:
            print(f"WARNING: no matching real file for {gen_name} → expected {real_name}")

    del clap  # Free memory
    torch.cuda.empty_cache()
    return scores

def get_wav2vec_embeddings(folder, model, processor, device="cuda"):
    
    embeddings = {}
    all_files = [f for f in os.listdir(folder) if f.endswith(".wav")]

    for fname in tqdm(all_files, desc=f"Extracting Wav2Vec2 from {folder}"):
        path = os.path.join(folder, fname)
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs.to(device))
        emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings[fname] = emb
    return embeddings

# Compute cosine/L1/L2 similarities
def compute_wav2vec_similarity(real_folder, generated_folder):
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()

    print("Extracting Wav2Vec2 embeddings for real audio...")
    real_embs = get_wav2vec_embeddings(real_folder, model, processor, device)
    print("Extracting Wav2Vec2 embeddings for generated audio...")
    gen_embs = get_wav2vec_embeddings(generated_folder, model, processor, device)

    scores = {
        'cosine': {},
        'l2': {},
        'l1': {}
    }

    for gen_name, gen_emb in gen_embs.items():
        real_name = get_real_name(gen_name)
        if real_name in real_embs:
            real_emb = real_embs[real_name]
            cosine = np.dot(gen_emb, real_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(real_emb))
            l2 = np.linalg.norm(gen_emb - real_emb)
            l1 = np.sum(np.abs(gen_emb - real_emb))

            scores['cosine'][gen_name] = cosine
            scores['l2'][gen_name] = l2
            scores['l1'][gen_name] = l1
        else:
            print(f"⚠️ Missing match for: {gen_name} → {real_name}")

    del model, processor  # Free memory
    torch.cuda.empty_cache()
    return scores


def get_spk_embs(folder):
    from speechbrain.pretrained import EncoderClassifier
    embs = {}
    spk_clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda" if torch.cuda.is_available() else "cpu"})
    for fname in tqdm(os.listdir(folder)):
        if not fname.endswith(".wav"): continue
        wav, _ = load_and_normalize(os.path.join(folder, fname))
        embs[fname] = spk_clf.encode_batch(torch.from_numpy(wav).unsqueeze(0).to(spk_clf.device)).squeeze().cpu().numpy()
    return embs


def load_and_normalize(path):
    from scipy.io import wavfile
    sr, wav = wavfile.read(path)
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav.astype(np.float32) - 128) / 128.0
    return wav, sr


def normalize_to_int16(wav):
    wav = wav / np.max(np.abs(wav))  # [-1, 1]
    return (wav * 32767).astype(np.int16)

def prepare_for_metric(wav):
    if isinstance(wav, torch.Tensor):
        wav = wav.squeeze().cpu().numpy()
    elif isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.squeeze()
    return wav.astype(np.float32)

def compute_stoi(ref_wav, gen_wav, sr=16000):
    from pystoi import stoi
    ref = prepare_for_metric(ref_wav)
    gen = prepare_for_metric(gen_wav)
    return stoi(ref, gen, sr, extended=False)


def transcribe(audio_path, model):
    result = model.transcribe(
        audio_path,
        fp16=torch.cuda.is_available(),   
        verbose=False,                    
        condition_on_previous_text=False, 
        word_timestamps=False          
    )
    return result['text'].strip().lower()

def compute_eer(spk_real_embs, spk_gen_embs):
    """
    Evaluates the cosine similarity between real and generated speaker embeddings.
    Returns the EER value.
    """
    y_true = []
    scores = []

    for gen_name, gen_emb in spk_gen_embs.items():
        real_name = get_real_name(gen_name)
        if real_name not in spk_real_embs:
            continue
        real_emb = spk_real_embs[real_name]

        # Positive samples
        sim_pos = np.dot(gen_emb, real_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(real_emb))
        scores.append(sim_pos)
        y_true.append(1)

        # Negative samples
        for neg_name, neg_emb in spk_real_embs.items():
            if neg_name == real_name:
                continue
            sim_neg = np.dot(gen_emb, neg_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(neg_emb))
            scores.append(sim_neg)
            y_true.append(0)

    # EER Evaluation
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer

def evaluate_all(real_folder, gen_folder):
    # Pre-estrazioni
    print("Extracting CLAP embeddings and computing similarities...")
    claps = compute_clap_similarity(real_folder, gen_folder)

    print("Extracting Wav2Vec2 embeddings and computing similarities...")
    w2v = compute_wav2vec_similarity(real_folder, gen_folder)

    print("Extracting speaker embeddings...")
    spk_real = get_spk_embs(real_folder)

    print("Extracting speaker embeddings for generated audio...")
    spk_gen = get_spk_embs(gen_folder)

    asr_model = whisper.load_model("base", device='cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    for gen_name in tqdm(w2v['cosine'], desc='Evaluating...'):  # assume chiave esiste
        real_name = get_real_name(gen_name)
        if real_name not in spk_real: 
            print(f"WARNING: No speaker embedding for {real_name}, skipping.")
            continue

        try:
            # Audio files
            ref_p = os.path.join(real_folder, real_name)
            gen_p = os.path.join(gen_folder, gen_name)
            sr = 16000  # Default sampling rate
            ref_wav, _ = load_and_normalize(ref_p)
            gen_wav, _ = load_and_normalize(gen_p)

            res = {}
            # CLAP metrics (cosine, l2, l1)
            res['clap_cosine'] = claps['cosine'].get(gen_name, np.nan)
            res['clap_l2'] = claps['l2'].get(gen_name, np.nan)
            res['clap_l1'] = claps['l1'].get(gen_name, np.nan)

            # Wav2Vec2 metrics
            res['wav_cosine'] = w2v['cosine'].get(gen_name, np.nan)
            res['wav_l2'] = w2v['l2'].get(gen_name, np.nan)
            res['wav_l1'] = w2v['l1'].get(gen_name, np.nan)

            # Speaker embedding distance (Euclidean)
            res['spk_dist'] = np.linalg.norm(spk_real[real_name] - spk_gen.get(gen_name, spk_real[real_name]))

            # Acoustic similarity metrics
            res['stoi'] = compute_stoi(ref_wav, gen_wav, sr)

            try:
                gen_txt = transcribe(gen_p, asr_model)
                real_txt = transcribe(ref_p, asr_model)
                res['wer'] = wer(real_txt, gen_txt)
            except Exception as e:
                print(f"⚠️ ASR failed on {gen_name}: {e}")
                res['wer'] = np.nan
        except Exception as e_main:
            print(f"⚠️ Error processing {gen_name}: {e_main}")
            # Set default NaNs for all expected fields
            for key in ['clap_cosine', 'clap_l2', 'clap_l1', 'wav_cosine', 'wav_l2', 'wav_l1',
                        'spk_dist', 'stoi', 'wer']:
                res[key] = np.nan
        results[gen_name] = res
    return results
