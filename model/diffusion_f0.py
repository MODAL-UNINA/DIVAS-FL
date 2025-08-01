import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import BaseModule
from model.diffusion_module import *
from math import sqrt

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer 

class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, dim_base): 
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(dim_base, residual_channels)
    self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, conditioner, x_mask):
    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step

    conditioner = self.conditioner_projection(conditioner)
    y = self.dilated_conv(y*x_mask) + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y*x_mask)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class GradLogPEstimator(BaseModule):
    def __init__(self, dim_base, dim_cond, res_layer=30, res_ch=64, dilation_cycle=10):
        super(GradLogPEstimator, self).__init__()

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4),
                                       Mish(), 
                                       torch.nn.Linear(dim_base * 4, dim_base), 
                                       Mish())

        cond_total = dim_base + 256 + 128
        self.cond_block = torch.nn.Sequential(Conv1d(cond_total, 4 * dim_cond, 1),
                                              Mish(),
                                              Conv1d(4 * dim_cond, dim_cond, 1), 
                                              Mish())

        self.input_projection = torch.nn.Sequential(Conv1d(1, res_ch, 1), Mish()) 
        self.residual_layers = nn.ModuleList([
            ResidualBlock(dim_cond, res_ch, 2 ** (i % dilation_cycle), dim_base)
            for i in range(res_layer)
        ])
        self.skip_projection = torch.nn.Sequential(Conv1d(res_ch, res_ch, 1), Mish())
        self.output_projection = Conv1d(res_ch, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, x_mask, f0, spk, t):
        condition = self.time_pos_emb(t) 
        t = self.mlp(condition)
        x = self.input_projection(x) * x_mask 

        condition = torch.cat([f0, condition.unsqueeze(-1).expand(-1, -1, f0.size(2)), spk.expand(-1, -1, f0.size(2))], 1)
        condition = self.cond_block(condition)*x_mask  

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, t, condition, x_mask)
            skip = skip_connection * x_mask if skip is None else (skip_connection + skip) * x_mask

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x) * x_mask
        x = self.output_projection(x) * x_mask
        
        return x 

    @torch.no_grad()
    def infer(self, x, x_mask, f0, spk, t):
        condition = self.time_pos_emb(t) 
        t = self.mlp(condition)
        x = self.input_projection(x) * x_mask 

        condition = torch.cat([f0, condition.unsqueeze(-1).expand(-1, -1, f0.size(2)), spk.expand(-1, -1, f0.size(2))], 1)
        condition = self.cond_block(condition)*x_mask  

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, t, condition, x_mask) 
            skip = skip_connection * x_mask if skip is None else (skip_connection + skip) * x_mask
 
        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x) * x_mask
        x = self.output_projection(x) * x_mask
        
        return x

class Diffusion(BaseModule):
    def __init__(self, n_feats, dim, dim_spk, beta_min, beta_max):
        super(Diffusion, self).__init__()
        self.estimator_f0 = GradLogPEstimator(dim, dim_spk)

        self.n_feats = n_feats
        self.dim_unet = dim
        self.dim_spk = dim_spk
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, t):
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        return beta

    def get_gamma(self, s, t, p=1.0, use_torch=False):
        beta_integral = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (t + s)
        beta_integral *= (t - s)
        if use_torch:
            gamma = torch.exp(-0.5 * p * beta_integral).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = math.exp(-0.5 * p * beta_integral)
        return gamma

    def get_mu(self, s, t):
        a = self.get_gamma(s, t)
        b = 1.0 - self.get_gamma(0, s, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_nu(self, s, t):
        a = self.get_gamma(0, s)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_sigma(self, s, t):
        a = 1.0 - self.get_gamma(0, s, p=2.0)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return math.sqrt(a * b / c)

    def compute_diffused_z_pr(self, x0, mask, z_pr, t, use_torch=False): 
        x0_weight = self.get_gamma(0, t, use_torch=use_torch)  
        z_pr_weight = 1.0 - x0_weight
        xt_z_pr = x0 * x0_weight + z_pr * z_pr_weight
        return xt_z_pr * mask 

    # def compute_diffused_z_pr(self, x0, mask, z_pr, t_values, use_torch=False): 
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     diff_values = []  # Lista per memorizzare le differenze tra i passi
    #     t_values = np.linspace(0, 1, num=10)
    #     # Inizializza il valore precedente (prima di t=0, prendi x0)
    #     previous_xt = x0

    #     for t in t_values:
    #         # Calcola il peso di gamma
    #         x0_weight = self.get_gamma(0, t, use_torch=use_torch)  
    #         z_pr_weight = 1.0 - x0_weight
    #         xt_z_pr = x0 * x0_weight + z_pr * z_pr_weight

    #         # Calcola la differenza tra xt_z_pr e il valore precedente (xt)
    #         diff_t = torch.abs(xt_z_pr - previous_xt).mean().item()  # Media per ottenere un singolo valore di differenza

    #         # Aggiungi la differenza alla lista
    #         diff_values.append(diff_t)

    #         # Aggiorna il valore precedente per il prossimo passo
    #         previous_xt = xt_z_pr

    #         xt_z_pr_2d = xt_z_pr.squeeze().detach().cpu()
    #         # Plot per xt_z_pr
    #         plt.figure(figsize=(4, 4))
    #         plt.plot(np.linspace(0, 1, len(xt_z_pr_2d)), xt_z_pr_2d)  # Usando np.linspace per il tempo
    #         plt.axis('off')
    #         plt.gca().set_facecolor('white')
    #         plt.savefig(f'/home/modal-workbench/Projects/Tesisti/Donato/Audio/Federated_gen/plot_paper/pitch_diffusion_result_{t}.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=False)
    #         plt.close()

    #     return xt_z_pr * mask  # Restituisce l'output finale
    
    def forward_diffusion(self, x0, mask, src_out, t):
        xt_src = self.compute_diffused_z_pr(x0, mask, src_out, t, use_torch=True)
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt_src = xt_src + z * torch.sqrt(variance)

        return xt_src * mask, z * mask

    @torch.no_grad()
    def reverse(self, z, mask, y_hat, z_f0, spk, ts):
        h = 1.0 / ts
        xt = z * mask
        
        for i in range(ts):
            t = 1.0 - i * h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t) 
            
            kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
            kappa /= (self.get_gamma(0, t) * beta_t * h)
            kappa -= 1.0
            omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
            omega += self.get_mu(t - h, t)
            omega -= (0.5 * beta_t * h + 1.0)
            sigma = self.get_sigma(t - h, t)  

            dxt = (y_hat - xt) * (0.5 * beta_t * h + omega) 
            dxt -= (self.estimator_f0.infer(xt, mask, z_f0, spk, time)) * (1.0 + kappa) * (beta_t * h)            
            dxt += torch.randn_like(z, device=z.device) * sigma 
            xt = (xt - dxt) * mask

        return xt

    # @torch.no_grad()
    # def reverse(self, z, mask, y_hat, z_f0, spk, ts):
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     h = 1.0 / ts
    #     xt = z * mask
        
    #     diff_values = []  # Lista per memorizzare le differenze tra i passi
        
    #     # Loop per il processo inverso
    #     for i in range(ts):
    #         t = 1.0 - i * h
    #         time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
    #         beta_t = self.get_beta(t) 
            
    #         kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
    #         kappa /= (self.get_gamma(0, t) * beta_t * h)
    #         kappa -= 1.0
            
    #         omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
    #         omega += self.get_mu(t - h, t)
    #         omega -= (0.5 * beta_t * h + 1.0)
            
    #         sigma = self.get_sigma(t - h, t)  

    #         # Calcola il cambiamento in xt
    #         dxt = (y_hat - xt) * (0.5 * beta_t * h + omega) 
    #         dxt -= (self.estimator_f0.infer(xt, mask, z_f0, spk, time)) * (1.0 + kappa) * (beta_t * h)            
    #         dxt += torch.randn_like(z, device=z.device) * sigma 
            
    #         # Aggiorna xt
    #         xt_z_pr_2d = xt.squeeze().detach().cpu()
    #         # Plot per xt_z_pr
    #         plt.figure(figsize=(4, 4))
    #         plt.plot(np.linspace(0, 1, len(xt_z_pr_2d)), xt_z_pr_2d)  # Usando np.linspace per il tempo
    #         plt.axis('off')
    #         plt.gca().set_facecolor('white')
    #         plt.savefig(f'/home/modal-workbench/Projects/Tesisti/Donato/Audio/Federated_gen/plot_paper/reverse_result_{t}.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=False)
    #         plt.close()
    #         xt = (xt - dxt) * mask

    #         # Calcola la differenza tra il passo corrente e il passo precedente
    #         diff_t = torch.abs(dxt).mean().item()  # Media della differenza per il passo corrente
    #         diff_values.append(diff_t)

    #     return xt  # Restituisce il risultato finale del processo reverse

    def compute_loss(self, x0, mask, x0_hat, spk, f0, t): 
        xt, z = self.forward_diffusion(x0, mask, x0_hat, t)
        z_estimation = self.estimator_f0(xt, mask, f0, spk, t)
        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((z_estimation + z) ** 2) / (torch.sum(mask))

        return loss

    def compute_t(self, x0, mask, x0_hat, f0, spk, offset=1e-5):
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        return self.compute_loss(x0, mask, x0_hat, spk, f0, t)
