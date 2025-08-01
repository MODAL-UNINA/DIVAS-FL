import math
import torch
import random
import numpy as np
from torch.nn import functional as F

from model.base import BaseModule
from model.diffusion_module import * 


class GradLogPEstimator(BaseModule):
    def __init__(self, dim_base, dim_cond, dim_mults=(1, 2, 4)):
        super(GradLogPEstimator, self).__init__()

        dims = [2 + dim_cond, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4),
                                       Mish(), torch.nn.Linear(dim_base * 4, dim_base))
        cond_total = dim_base + 256
        self.cond_block = torch.nn.Sequential(torch.nn.Linear(cond_total, 4 * dim_cond),
                                              Mish(), torch.nn.Linear(4 * dim_cond, dim_cond))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([]) 
        num_resolutions = len(in_out)  

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]  
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim_base),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in)]))
             
        self.m_final_block = Block(dim_base, dim_base)
        self.m_final_conv = torch.nn.Conv2d(dim_base, 1, 1)
        
        self.z_final_block = Block(dim_base, dim_base)
        self.z_final_conv = torch.nn.Conv2d(dim_base, 1, 1)

    def forward(self, x, x_mask, enc_out, spk, t):
        condition = self.time_pos_emb(t)  
        t = self.mlp(condition)

        x = torch.stack([enc_out, x], 1)
        x_mask = x_mask.unsqueeze(1)

        condition = torch.cat([condition, spk.squeeze(2)], 1) 
        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)  

        condition = torch.cat(x.shape[2] * [condition], 2)  
        condition = torch.cat(x.shape[3] * [condition], 3)
        x = torch.cat([x, condition], 1)

        hiddens = []
        masks = [x_mask]

        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)
        
        m_x = self.m_final_block(x, x_mask)
        m_output = self.m_final_conv(m_x * x_mask)
        
        z_x = self.z_final_block(x, x_mask)
        z_output = self.z_final_conv(z_x * x_mask)

        return (m_output * x_mask).squeeze(1), (z_output * x_mask).squeeze(1)


class Diffusion(BaseModule):
    def __init__(self, n_feats, dim_unet, dim_spk, beta_min, beta_max):
        super(Diffusion, self).__init__()
        self.estimator = GradLogPEstimator(dim_unet, dim_spk)

        self.n_feats = n_feats
        self.dim_unet = dim_unet
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
    #     t_values = np.linspace(0, 1, num=10)  # Genera 10 valori di t tra 0 e 1

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
    #         plt.imshow(xt_z_pr_2d, aspect='auto', origin='lower')
    #         plt.axis('off')
    #         plt.savefig(f'/home/modal-workbench/Projects/Tesisti/Donato/Audio/Federated_gen/plot_paper/mel_diffusion_result_{t}.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    #         plt.close()

    #     return xt_z_pr * mask  # Restituisce l'output finale

    @torch.no_grad()
    def reverse(self, z, mask, z_pr, spk, ts):
        import matplotlib.pyplot as plt
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

            dxt = (z_pr - xt) * (0.5 * beta_t * h + omega) 
            tmp, dxt_ = self.estimator(xt, mask, z_pr, spk, time) 
            dxt -= dxt_ * (1.0 + kappa) * (beta_t * h)
            dxt += torch.randn_like(z, device=z.device) * sigma
            # # Aggiorna xt
            # xt_z_pr_2d = xt.squeeze().detach().cpu()
            # # Plot per xt_z_pr
            # plt.figure(figsize=(4, 4))
            # plt.imshow(xt_z_pr_2d, aspect='auto', origin='lower')
            # plt.axis('off')
            # plt.savefig(f'/home/modal-workbench/Projects/Tesisti/Donato/Audio/Federated_gen/plot_paper/mel_reverse_result_{t}.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
            # plt.close()
            xt = (xt - dxt) * mask
            
        return xt 
    
    @torch.no_grad()
    def forward(self, z, mask, enc_out, spk, n_timesteps, mode): 
        return self.reverse_diffusion(z, mask, enc_out, spk, n_timesteps, mode)

    def random_masking(self, xt, num, frame): 
        xt_mask = torch.ones_like(xt)
        x0_mask = torch.ones_like(xt)
        for _ in range(num):
            idx = random.randint(0, xt.size(1)-frame)
            xt[:, idx:idx+frame, :] = 0
            xt_mask[:, idx:idx+frame, :] = 0
        x0_mask -= xt_mask  
        
        return xt, xt_mask, x0_mask


    # def compute_diffused_z_pr(self, x0, mask, z_pr, t, use_torch=False):
    #     x0_weight = self.get_gamma(0, t, use_torch=use_torch)  
    #     z_pr_weight = 1.0 - x0_weight
    #     xt_z_pr = x0 * x0_weight + z_pr * z_pr_weight
    #     return xt_z_pr * mask  
    
    
    def forward_diffusion(self, x0, mask, enc_out, t):
        xt = self.compute_diffused_z_pr(x0, mask, enc_out, t, use_torch=True)
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = xt + z * torch.sqrt(variance)
    
        return xt * mask, z * mask 

    def compute_loss(self, x0, mask, enc_out, spk, t):
        xt, z = self.forward_diffusion(x0, mask, enc_out, t) 
        masked_xt, xt_mask, x0_mask = self.random_masking(xt, num=4, frame=8) 
     
        m_estimation, z_estimation = self.estimator(masked_xt, mask, enc_out, spk, t) 
        m_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        diff_loss = torch.sum((z_estimation*xt_mask + z) ** 2) / (torch.sum(mask) * self.n_feats)
        recon_loss = F.l1_loss(x0*x0_mask, m_estimation*x0_mask)

        return diff_loss, recon_loss

    def compute_t(self, x0, mask, enc_out, spk, offset=1e-5):
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        return self.compute_loss(x0, mask, enc_out, spk, t)
