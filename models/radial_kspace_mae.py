from functools import partial

import torch
import torch.nn as nn
import numpy as np
import math
from models.radial_kspace_transformer import RadialKspaceEmbedCombinedLinear, Block

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class RadialKspaceMaskedAutoencoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, num_spokes=64, spoke_length=64,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_real_imaginary=True,
                 loss_type="MSE" # other option is HDR loss (see k-gin paper or NeRF in dark or noise2noise)
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.token_embed = RadialKspaceEmbedCombinedLinear(num_spokes=num_spokes, 
                                                           spoke_length=spoke_length, 
                                                           embed_dim=embed_dim, 
                                                           use_real_imaginary=use_real_imaginary,
                                                           )
        self.num_tokens = num_spokes 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                #   qk_scale=None, # not in the attention block we are using 
                  norm_layer=norm_layer,
                  )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, 
                #   qk_scale=None, 
                  norm_layer=norm_layer,
                  )
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, spoke_length * 2, bias=True) # decoder to patch, both real and imaginary
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.loss_type = loss_type
        print("Loss type: ", self.loss_type)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = positionalencoding1d(self.pos_embed.shape[-1], self.num_tokens + 1)
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        decoder_pos_embed = positionalencoding1d(self.decoder_pos_embed.shape[-1], self.num_tokens + 1)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.token_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    #     h = w = imgs.shape[2] // p
    #     x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    #     return x

    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1]**.5)
    #     assert h * w == x.shape[1]
        
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed tokens
        x = self.token_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, x, pred, mask):
        """
        x: [N, num_spokes, spoke_length] it's a complex tensor
        pred: [N, num_spokes, spoke_length*2] predicting real and imag for each masked spoke
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # generating target
        real = torch.real(x) # [N, num_spokes, spoke_len]
        imag = torch.imag(x) # [N, num_spokes, spoke_len]
        mag = torch.abs(x)

        # pred imag and real
        pred_real = pred[:, :, :x.shape[-1]]
        pred_imag = pred[:, :, x.shape[-1]:]
        
        mag_scaling = torch.cat([mag, mag], dim=-1)
        target = torch.cat([real, imag], dim=-1)

        if self.norm_pix_loss:
            real_mean = real.mean(dim=-1, keepdim=True)
            imag_mean = imag.mean(dim=-1, keepdim=True)
            real_var = real.var(dim=-1, keepdim=True)
            imag_var = imag.var(dim=-1, keepdim=True)
            real_target = (real - real_mean) / (real_var + 1.e-6)**.5
            imag_target = (imag - imag_mean) / (imag_var + 1.e-6)**.5
            
            target = torch.cat([real_target, imag_target], dim=-1)

        if self.loss_type == "MSE":
            loss = (pred - target) ** 2
        elif self.loss_type == "HDR":
            loss = ( (pred - target) / (pred.detach() + 1e-8) )**2
        elif self.loss_type == "HDR_static":
            loss = ( (pred - target) / (mag_scaling + 1e-8) )**2
        elif self.loss_type == "HDR_static_phase_preserving":
            pred_phase = torch.atan2(pred_imag, pred_real)
            target_phase = torch.atan2(imag, real)
            loss = ( (pred - target) / (mag_scaling + 1e-8) )**2 # 896 at dim 2
            phase_preserving_loss = (pred_phase - target_phase)**2 # 448 ad dim 2

            # loss = ( (pred - target) / (mag_scaling + 1e-8) )**2 + (pred_phase - target_phase)**2
        elif self.loss_type == "arda_loss":
            pred = torch.complex(pred_real, pred_imag)
            target = x

            loss = torch.linalg.norm(target.flatten() - pred.flatten(), ord=2) / torch.linalg.norm(target.flatten(), ord=2) + torch.linalg.norm(target.flatten() - pred.flatten(), ord=1) / torch.linalg.norm(target.flatten(), ord=1)

        elif self.loss_type == "arda_loss_l2":
            pred = torch.complex(pred_real, pred_imag)
            target = x

            loss = torch.linalg.norm(target.flatten() - pred.flatten(), ord=2) / torch.linalg.norm(target.flatten(), ord=2) 


        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if self.loss_type == "HDR_static_phase_preserving":
            loss = loss + phase_preserving_loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
