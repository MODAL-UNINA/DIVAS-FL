import torch
from torch import nn
from torch.nn import functional as F 

from model.base import BaseModule
from model.diffusion_mel import Diffusion as Mel_Diffusion
from model.diffusion_f0 import Diffusion as F0_Diffusion
from model.styleencoder import StyleEncoder  

import transformers

from module.modules import *
from module.utils import * 


class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=12): 
        super().__init__() 
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer
        
    @torch.no_grad()
    def forward(self, x): 
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]    
        
        return y.permute((0, 2, 1))    

class Encoder(nn.Module):
    def __init__(self,
                in_channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                mel_size=80,
                gin_channels=0,
                p_dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, p_dropout=p_dropout)
        self.proj = nn.Conv1d(hidden_channels, mel_size, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x * x_mask) * x_mask
        x = self.enc(x, x_mask, g=g)
        x = self.proj(x) * x_mask

        return x


class PlanarFlow(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.transforms = nn.ModuleList([PlanarTransform(dim) for k in range(K)])
    def forward(self, z, logdet=False):
        zK = z
        SLDJ = 0.
        for transform in self.transforms:
            out = transform(zK, logdet=logdet)
            if logdet:
                SLDJ += out[1]
                zK = out[0]
            else:
                zK = out
                
        if logdet:
            return zK, SLDJ
        return zK


class PlanarTransform(nn.Module):
    def __init__(self, dim=20):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.randn(()) * 0.01)
    def m(self, x):
        return -1 + torch.log(1 + torch.exp(x))
    def h(self, x):
        return torch.tanh(x)
    def h_prime(self, x):
        return 1 - torch.tanh(x) ** 2
    def forward(self, z, logdet=False):
        # z.size() = batch x dim
        u_dot_w = (self.u @ self.w.t()).view(())
        w_hat = self.w / torch.norm(self.w, p=2) # Unit vector in the direction of w
        u_hat = (self.m(u_dot_w) - u_dot_w) * (w_hat) + self.u # 1 x dim
        affine = z @ self.w.t() + self.b
        z_next = z + u_hat * self.h(affine) # batch x dim
        if logdet:
            psi = self.h_prime(affine) * self.w # batch x dim
            LDJ = -torch.log(torch.abs(psi @ u_hat.t() + 1) + 1e-8) # batch x 1
            return z_next, LDJ
        return z_next


class StyleVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, cond_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim if cond_dim is not None else 0
        self.conc_dim = latent_dim + cond_dim if cond_dim > 0 else latent_dim
        
        self.style_mu_shared = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.style_mu_male = nn.Linear(self.hidden_dim + self.cond_dim, self.latent_dim)
        self.style_mu_female = nn.Linear(self.hidden_dim + self.cond_dim, self.latent_dim)

        self.style_logvar_shared = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.style_logvar_male = nn.Linear(self.hidden_dim + self.cond_dim, self.latent_dim)
        self.style_logvar_female = nn.Linear(self.hidden_dim + self.cond_dim, self.latent_dim)

        # Planar flow layers
        self.flow = PlanarFlow(self.latent_dim, K=2)  # Number of planar transforms can be adjusted

        if self.cond_dim > 0:
            self.gender_cls = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.latent_dim // 2, 2)
            )
        
        # Mixture of Experts
        self.expert_1 = nn.Sequential(
            nn.Linear(self.conc_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.expert_2 = nn.Sequential(
            nn.Linear(self.conc_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(self.conc_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

        # Final projection layer
        self.final_projection = nn.Linear(self.hidden_dim, self.in_dim)

    def forward(self, x, cond=None, gender_labels=None):
        """
        Forward pass of the StyleVAE.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).
            cond (torch.Tensor): Condition tensor of shape (batch_size, cond_dim).
            gender_labels (torch.Tensor): Label tensor of shape (batch_size,).
        Returns:
            x_recon (torch.Tensor): Reconstructed input tensor.
            log_q0_zK - log_prior_zK (torch.Tensor): KL divergence term.
            gender_logits (torch.Tensor): Gender classification logits.
        """
        batch_size = x.size(0)


        shared_mu = self.style_mu_shared(x)
        shared_logvar = self.style_logvar_shared(x)
        if cond is not None:
            shared_mu = torch.cat((shared_mu, cond), dim=-1)
            shared_logvar = torch.cat((shared_logvar, cond), dim=-1)
            if gender_labels is not None:
                # Encoder based on gender labels
                mu = torch.zeros(batch_size, self.latent_dim, device=x.device)
                logvar = torch.zeros(batch_size, self.latent_dim, device=x.device)
                male_mask = (gender_labels == 1)
                female_mask = (gender_labels == 0)
                if male_mask.any():
                    mu[male_mask] = self.style_mu_male(shared_mu[male_mask])
                    logvar[male_mask] = self.style_logvar_male(shared_logvar[male_mask])
                if female_mask.any():
                    mu[female_mask] = self.style_mu_female(shared_mu[female_mask])
                    logvar[female_mask] = self.style_logvar_female(shared_logvar[female_mask])
            else:
                # Fallback 
                mu = self.style_mu_male(shared_mu)  # Male as default
                logvar = self.style_logvar_male(shared_logvar)
        else:
            # Fallback
            mu = self.style_mu_male(shared_mu)  # Use male as default
            logvar = self.style_logvar_male(shared_logvar)

        logvar = torch.clamp(logvar, min=-5, max=5)  # Clamp for stability

        # Reparameterization trick
        z0 = self.reparameterize(mu, logvar)

        # Apply planar flow
        zK, logdet = self.flow(z0, logdet=True)

        # Prior e posterior
        q0 = torch.distributions.normal.Normal(mu, (0.5 * logvar).exp())
        prior = torch.distributions.normal.Normal(0., 1.)
        log_prior_zK = prior.log_prob(zK).sum(-1)
        log_q0_z0 = q0.log_prob(z0).sum(-1)
        log_q0_zK = log_q0_z0 + logdet.view(-1)

        if cond is not None:
            # Classification
            gender_logits = self.gender_cls(zK)
            zK = torch.cat((zK, cond), dim=-1)
        else:
            gender_logits = None

        # Mixture of Experts decoder
        expert_1_out = self.expert_1(zK)
        expert_2_out = self.expert_2(zK)
        gate_weights = self.gate_network(zK)
        combined_expert = (gate_weights[:, 0:1] * expert_1_out + 
                          gate_weights[:, 1:2] * expert_2_out)
        
        # Reconstruction
        x_recon = self.final_projection(combined_expert)

        return x_recon.unsqueeze(-1), log_q0_zK - log_prior_zK, gender_logits.squeeze(-1) if gender_logits is not None else None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class SynthesizerTrn(nn.Module):
    def __init__(self, hidden_size, latent_dim):
        super().__init__()

        # Hyperparameters
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.gender_dim = 16  

        # Embedding layers
        self.emb_c = nn.Conv1d(1024, hidden_size, 1)
        self.emb_c_f0 = nn.Conv1d(1024, hidden_size, 1)
        self.emb_f0 = nn.Conv1d(1, hidden_size, kernel_size=9, stride=4, padding=4) 
        self.emb_norm_f0 = nn.Conv1d(1, hidden_size, 1)
        self.gender_emb = nn.Embedding(2, self.gender_dim)

        # Style VAE
        self.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256)
        self.style_vae = StyleVAE(in_dim=256, hidden_dim=self.hidden_size, latent_dim=self.latent_dim, cond_dim=self.gender_dim)
        
        # Content and F0 encoders
        self.mel_enc_c = Encoder(hidden_size, hidden_size, 5, 1, 8, 80, gin_channels=256, p_dropout=0)
        self.mel_enc_f = Encoder(hidden_size, hidden_size, 5, 1, 8, 80, gin_channels=256, p_dropout=0)
        self.f0_enc = Encoder(hidden_size, hidden_size, 5, 1, 8, 128, gin_channels=256, p_dropout=0)
        self.proj = nn.Conv1d(hidden_size, 1, 1)

    def forward(self, x_mel, w2v, norm_f0, f0, x_mask, f0_mask, gender=None):
        content = self.emb_c(w2v) 
        content_f = self.emb_c_f0(w2v)
        f0 = self.emb_f0(f0)  
        norm_f0 = self.emb_norm_f0(norm_f0)
        if gender is not None:
            gender_emb = self.gender_emb(gender)

        g = self.emb_g(x_mel, x_mask).unsqueeze(-1)
        g_recon, kl, gender_logits = self.style_vae(g.squeeze(-1), gender_emb, gender)

        y_cont = self.mel_enc_c(F.relu(content), x_mask, g=g_recon)
        y_f0 = self.mel_enc_f(F.relu(f0), x_mask, g=g_recon)
        y_mel = y_cont + y_f0

        content_f = F.interpolate(content_f, norm_f0.shape[-1])
        enc_f0 = self.f0_enc(F.relu(content_f+norm_f0), f0_mask, g=g_recon)
        y_f0_hat = self.proj(enc_f0)

        return g, y_mel, enc_f0, y_f0_hat, g_recon, kl, gender_logits

    def spk_embedding(self, mel, length):
        x_mask = torch.unsqueeze(commons.sequence_mask(length, mel.size(-1)), 1).to(mel.dtype) 
        
        return self.emb_g(mel, x_mask).unsqueeze(-1)

    def mel_predictor(self, w2v, x_mask, spk, pred_f0):
        content = self.emb_c(w2v) 
        pred_f0 = self.emb_f0(pred_f0) 

        y_cont = self.mel_enc_c(F.relu(content), x_mask, g=spk)
        y_f0 = self.mel_enc_f(F.relu(pred_f0), x_mask, g=spk)
        y_mel = y_cont + y_f0
        
        return y_mel
    
    def f0_predictor(self, w2v, x_f0_norm, f0_mask, fixed_noise, diversity_scale, gender=None):
        content_f = self.emb_c_f0(w2v)
        norm_f0 = self.emb_norm_f0(x_f0_norm)
        g_recon = self.sample_diverse_voice(fixed_noise, gender, diversity_scale, seed=None)  # Genera voce diversificata
        content_f = F.interpolate(content_f, norm_f0.shape[-1])
        
        enc_f0 = self.f0_enc(F.relu(content_f+norm_f0), f0_mask, g=g_recon)
        y_f0_hat = self.proj(enc_f0) 
    
        return g_recon, y_f0_hat, enc_f0, g_recon
    
    def sample_diverse_voice(self, noise, gender, diversity_scale, seed=None):
        """
        Sample a diverse voice embedding for a given gender

        Args:
            gender: 0 for female, 1 for male
            diversity_scale: Factor to increase diversity (1.0 = normal)
            seed: Seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Sample from the prior with increased diversity
        z_sample = noise * diversity_scale

        # Apply flow
        z_sample = self.style_vae.flow(z_sample, logdet=False)

        # Add gender embedding
        if gender is not None:
            gender_tensor = torch.tensor([gender], dtype=torch.long).to(z_sample.device)
            gender_emb = self.gender_emb(gender_tensor)

            # Apply flow
            z_sample = torch.cat([z_sample, gender_emb], dim=-1)
        
        expert_1_out = self.style_vae.expert_1(z_sample)
        expert_2_out = self.style_vae.expert_2(z_sample)
        gate = self.style_vae.gate_network(z_sample)
        combined = gate[:, 0:1] * expert_1_out + gate[:, 1:2] * expert_2_out

        g_recon = self.style_vae.final_projection(combined)

        return g_recon.unsqueeze(-1)


class DIVAS(BaseModule):
    def __init__(self, n_feats, spk_dim, dec_dim, beta_min, beta_max, hps):
        super(DIVAS, self).__init__()
        self.n_feats = n_feats
        self.spk_dim = spk_dim
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.hidden_size = hps.model.hidden_size
        self.latent_dim = hps.model.latent_dim

        self.encoder = SynthesizerTrn(self.hidden_size, self.latent_dim)
        self.f0_dec = F0_Diffusion(n_feats, 64, self.spk_dim, self.beta_min, self.beta_max)  
        self.mel_dec = Mel_Diffusion(n_feats, dec_dim, self.spk_dim, self.beta_min, self.beta_max)

    
    @torch.no_grad()
    def forward(self, x, w2v, norm_y_f0, f0_x, x_length, x_gender, n_timesteps, mode='ml'):
        """
        Complete forward pass with target voice generation
        """
        x_mask = sequence_mask(x_length, x.size(2)).unsqueeze(1).to(x.dtype) 
        f0_mask = sequence_mask(x_length*4, x.size(2)*4).unsqueeze(1).to(x.dtype)

        max_length = int(x_length.max())

        # SynthesizerTrn and Style VAE
        spk, y_mel, h_f0, y_f0_hat, spk_rec, kl, gender_logits = self.encoder(x, w2v, norm_y_f0, f0_x, x_mask, f0_mask, x_gender)

        # Diffusion F0  
        f0_mean_x = self.f0_dec.compute_diffused_z_pr(f0_x, f0_mask, y_f0_hat, 1.0)
        z_f0 = f0_mean_x * f0_mask
        z_f0 += torch.randn_like(z_f0, device=z_f0.device) 
        o_f0 = self.f0_dec.reverse(z_f0, f0_mask, y_f0_hat*f0_mask, h_f0*f0_mask, spk_rec, n_timesteps)
        
        # Diffusion Mel
        z_mel = self.mel_dec.compute_diffused_z_pr(x, x_mask, y_mel, 1.0) 
        z_mel += torch.randn_like(z_mel, device=z_mel.device)
        o_mel = self.mel_dec.reverse(z_mel, x_mask, y_mel, spk_rec, n_timesteps)

        # VAE loss
        beta = 0.1
        gamma = 0.5
        delta = 0.1

        spk_loss = torch.sum((spk - spk_rec) ** 2) / self.hidden_size
        kl_loss = kl.mean()
        gender_loss = F.cross_entropy(gender_logits, x_gender)
        
        # Diversity loss (avoid collapse)
        batch_size = spk_rec.size(0)
        if batch_size > 1:
            # Compute intra-batch diversity for same gender
            diversity_loss = 0.0
            for gender_val in [0, 1]:  # 0=female, 1=male
                gender_mask = (x_gender == gender_val)
                if gender_mask.sum() > 1:
                    same_gender_embeddings = spk_rec[gender_mask]
                    # Compute L2 distance between all pairs
                    pairwise_dist = torch.cdist(same_gender_embeddings.squeeze(-1), 
                                            same_gender_embeddings.squeeze(-1), p=2)
                    # We want to maximize the minimum distance (avoid collapse)
                    pairwise_dist = pairwise_dist + torch.eye(pairwise_dist.size(0), device=pairwise_dist.device) * 1e6
                    mean_min_dist = pairwise_dist.min(dim=1)[0].mean()
                    diversity_loss += torch.exp(-mean_min_dist)

            diversity_loss = diversity_loss * delta  # Weight of the diversity term
        else:
            diversity_loss = 0.0
        
        vae_loss = spk_loss + beta * kl_loss + gamma * gender_loss + diversity_loss

        return y_f0_hat, y_mel, o_f0, o_mel[:, :, :max_length], vae_loss
    
    def sample_diverse_voice(self, gender, diversity_scale=1.0, seed=None):
        """
        Sample a diverse voice for a given gender

        Args:
            gender: 0 for female, 1 for male
            diversity_scale: Factor to increase diversity (1.0 = normal)
            seed: Seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Sample from the prior with increased diversity
        latent_dim = self.encoder.latent_dim
        z_sample = torch.randn(1, latent_dim) * diversity_scale

        # Apply flow
        z_sample = self.encoder.style_vae.flow(z_sample, logdet=False)

        # Add gender embedding
        if gender is not None:
            gender_tensor = torch.tensor([gender], dtype=torch.long).to(z_sample.device)
            gender_emb = self.gender_emb(gender_tensor)
        
            z_sample = torch.cat([z_sample, gender_emb], dim=-1)
        
        # Use mixture of experts decoder
        expert_1_out = self.encoder.style_vae.expert_1(z_sample)
        expert_2_out = self.encoder.style_vae.expert_2(z_sample)
        gate = self.encoder.style_vae.gate_network(z_sample)
        combined = gate[:, 0:1] * expert_1_out + gate[:, 1:2] * expert_2_out
        
        style_embedding = self.encoder.style_vae.final_projection(combined)
        
        return style_embedding.unsqueeze(-1)
    
    def infer_vc(self, x, x_w2v, x_f0_norm, x_f0, x_length, diffpitch_ts, diffvoice_ts, fixed_noise, diversity_scale, gender=None):
        """
        Inference with generated target voice.

        Args:
            x: Mel spectrogram input
            x_w2v: Wav2Vec2 features
            x_f0_norm: Normalized F0
            x_f0: Original F0
            x_length: Length of each sequence in the batch
            diffpitch_ts: Number of timesteps for Diff-Pitch
            diffvoice_ts: Number of timesteps for Diff-Voice
            fixed_noise: Fixed noise for diversity sampling
            diversity_scale: Scale factor for diversity
            gender: Gender of the target voice
        """
        x_mask = sequence_mask(x_length, x.size(2)).unsqueeze(1).to(x.dtype)
        f0_mask = sequence_mask(x_length*4, x.size(2)*4).unsqueeze(1).to(x.dtype)

        # Predizione F0 con target voice generato
        spk, y_f0_hat, enc_f0, spk_rec = self.encoder.f0_predictor(x_w2v, x_f0_norm, f0_mask, fixed_noise, diversity_scale, gender)
        
        # Diff-Pitch
        z_f0 = self.f0_dec.compute_diffused_z_pr(x_f0, f0_mask, y_f0_hat, 1.0) 
        z_f0 += torch.randn_like(z_f0, device=z_f0.device)
        pred_f0 = self.f0_dec.reverse(z_f0, f0_mask, y_f0_hat*f0_mask, enc_f0*f0_mask, spk_rec, ts=diffpitch_ts)
        f0_zeros_mask = (x_f0 == 0)
        pred_f0[f0_zeros_mask.expand_as(pred_f0)] = 0 

        # Diff-Voice
        y_mel = self.encoder.mel_predictor(x_w2v, x_mask, spk_rec, pred_f0)
        z_mel = self.mel_dec.compute_diffused_z_pr(x, x_mask, y_mel, 1.0) 
        z_mel += torch.randn_like(z_mel, device=z_mel.device) 
        o_mel = self.mel_dec.reverse(z_mel, x_mask, y_mel, spk_rec, ts=diffvoice_ts)
    
        return o_mel[:, :, :x_length]

    def compute_loss(self, x, w2v_x, norm_f0_x, f0_x, x_length, x_gender):
        """
        Compute the complete loss for the model.
        This includes the VAE loss, Diff-Pitch loss, and Diff-Voice loss.
        Args:
            x: Mel spectrogram input
            w2v_x: Wav2Vec2 features
            norm_f0_x: Normalized F0
            f0_x: Original F0
            x_length: Length of each sequence in the batch
            x_gender: Gender of the input
        """ 
        x_mask = sequence_mask(x_length, x.size(2)).unsqueeze(1).to(x.dtype)
        f0_mask = sequence_mask(x_length*4, x.size(2)*4).unsqueeze(1).to(x.dtype)
    
        # spk, y_mel, y_f0, y_f0_hat, spk_rec, style_logvar, style_mu, gender_logits = self.encoder(x, w2v_x, norm_f0_x, f0_x, x_mask, f0_mask, x_gender)
        spk, y_mel, y_f0, y_f0_hat, spk_rec, kl, gender_logits = self.encoder(x, w2v_x, norm_f0_x, f0_x, x_mask, f0_mask, x_gender)

        # Calcola le losses
        beta = 0.1
        gamma = 0.5
        delta = 0.1
        f0_loss = torch.sum(torch.abs(f0_x - y_f0_hat)*f0_mask) / (torch.sum(f0_mask)) 
        mel_loss = torch.sum(torch.abs(x - y_mel)*x_mask) / (torch.sum(x_mask) * self.n_feats)
        spk_loss = torch.sum((spk - spk_rec) ** 2) / self.hidden_size
        kl_loss = kl.mean()
        gender_loss = F.cross_entropy(gender_logits, x_gender)

        # Diversity loss (avoid collapse)
        batch_size = spk_rec.size(0)
        if batch_size > 1:
            # Calcola diversità intra-batch per stesso genere
            diversity_loss = 0.0
            for gender_val in [0, 1]:  # 0=female, 1=male
                gender_mask = (x_gender == gender_val)
                if gender_mask.sum() > 1:
                    same_gender_embeddings = spk_rec[gender_mask]
                    # Calcola distanza L2 tra tutti i pair
                    pairwise_dist = torch.cdist(same_gender_embeddings.squeeze(-1), 
                                            same_gender_embeddings.squeeze(-1), p=2)
                    # Vogliamo massimizzare la distanza minima (evitare collapse)
                    pairwise_dist = pairwise_dist + torch.eye(pairwise_dist.size(0), device=pairwise_dist.device) * 1e6
                    mean_min_dist = pairwise_dist.min(dim=1)[0].mean()
                    diversity_loss += torch.exp(-mean_min_dist)
            
            diversity_loss = diversity_loss * delta  # Weight del termine di diversità
        else:
            diversity_loss = 0.0
        
        vae_loss = spk_loss + beta * kl_loss + gamma * gender_loss + diversity_loss
        f0_diff_loss = self.f0_dec.compute_t(f0_x, f0_mask, y_f0_hat, y_f0, spk)
        mel_diff_loss, mel_recon_loss  = self.mel_dec.compute_t(x, x_mask, y_mel, spk)

        return mel_diff_loss, mel_recon_loss, f0_diff_loss, mel_loss, f0_loss, vae_loss

