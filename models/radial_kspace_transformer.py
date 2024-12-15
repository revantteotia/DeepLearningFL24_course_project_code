""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange

# from baselines.ViT.helpers import load_pretrained
from .weight_init import trunc_normal_
from .helpers import to_2tuple
import math

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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        b, n, _, h = *x.shape, self.num_heads

        # self.save_output(x)
        # x.register_hook(self.save_output_grad)

        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook=False):
        x = x + self.attn(self.norm1(x), register_hook=register_hook)
        x = x + self.mlp(self.norm2(x))
        return x

#######################################################
# Kspace transformer stuff
#######################################################

# class RadialKspaceEmbedLinear(nn.Module):
#     """ Radial Kspace to Tokens Embedding
#         Takes input = batch of radial kspace torch tensors [bs, num_spokes, spoke_length] -> it is a complex tensor
#         Returns: [bs, num_tokens_per_kspace, token_dim]
#     """
#     def __init__(self, 
#                  num_spokes=64, spoke_length=64,
#                  embed_dim=768, 
#                  use_real_imaginary=True):
#         super().__init__()
#         self.num_spokes = num_spokes
#         self.spoke_length = spoke_length
#         self.embed_dim = embed_dim

#         # NOTE: removing patch embedding for now. why?: coz not in ViT and we may do global norm in the data generator
#         print("No pre layer norm before spoke embedding")
#         # pre norm before linear projection
#         # self.mag_norm_layer = nn.LayerNorm(W)
#         # self.phase_norm_layer = nn.LayerNorm(W)
#         # check line 118 and nlp example here: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

#         self.channel_1_proj = nn.Linear(spoke_length, embed_dim)  # channel 1 can be magnitude or real based on experiment
#         self.channel_2_proj = nn.Linear(spoke_length, embed_dim)  # channel 2 can be phase or imag based on experiment

#         # NOTE: post norm?? I guess not needed as both outputs are by nn linear and already normalized
#         # Also, attn blocks have their own pre norms

#     def forward(self, x):
#         # NOTE: 2 channels are decided by dataloader
#         ch1, ch2 = x
        
#         # NOTE: no need now, separate channels are not complex
#         # lets first convert to complex64 because learnable matrix are float not double
#         ch1 = ch1.to(torch.float32) # converting from double to float
#         ch2 = ch2.to(torch.float32)

#         B, num_spokes, spoke_length = ch1.shape # [bs, num_spokes, spoke_length]
        
#         # FIXME look at relaxing size constraints
#         assert num_spokes == self.num_spokes and spoke_length == self.spoke_length, \
#             f"Input kspace size ({num_spokes}*{spoke_length}) doesn't match model ({self.num_spokes}*{self.spoke_length})."
        
#         # passing thru linear layer
#         ch1 = rearrange(ch1, "bs num_spokes spoke_length -> (bs num_spokes) spoke_length")
#         ch2 = rearrange(ch2, "bs num_spokes spoke_length -> (bs num_spokes) spoke_length")
        
#         ch1_embedding = self.channel_1_proj(ch1) # shape [(bs num_spokes) embed_dim]
#         ch2_embedding = self.channel_2_proj(ch2) # shape [(bs num_spokes) embed_dim]
        
#         # adding two embeddings
#         embedding = ch1_embedding + ch2_embedding
#         embedding = rearrange(embedding, "(bs sq_len) embed_dim -> bs sq_len embed_dim", bs=B, sq_len=num_spokes)

#         # I guess no need to normalize again as Transformer attn layer does that anyway
#         # also, both are outputs of nn.linear so should be already normailzed
#         return embedding

# class RadialKspaceTransformer(nn.Module):
#     """ Radial Kspace Transformer
#     """
#     def __init__(self, num_spokes=64, spoke_length=64, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm,
#                  use_real_imaginary=True,
#                  learned_pos_encoding=False):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
#         self.radial_kspace_embedding = RadialKspaceEmbedLinear(num_spokes=num_spokes, spoke_length=spoke_length, embed_dim=embed_dim, use_real_imaginary=use_real_imaginary)
#         num_tokens = num_spokes 

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         if learned_pos_encoding:
#             self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
#         else:
#             self.pos_embed = positionalencoding1d(embed_dim, num_tokens + 1)

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                 drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)

#         # Classifier head
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

#     def forward(self, x, register_hook=False):
#         ch1, ch2 = x
#         B = ch1.shape[0]
#         x = self.radial_kspace_embedding(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         # print("x.shape just before transformer blocks: ", x.shape) # [bs, 224 + 1, 768]

#         for blk in self.blocks:
#             x = blk(x, register_hook=register_hook)

#         x = self.norm(x)
#         x = x[:, 0]
#         x = self.head(x)
#         return x

class RadialKspaceEmbedCombinedLinear(nn.Module):
    """ Radial Kspace to Tokens Embedding
        Takes input = batch of radial kspace torch tensors [bs, num_spokes, spoke_length] -> it is a complex tensor
        Returns: [bs, num_tokens_per_kspace, token_dim]
    """
    def __init__(self, 
                 num_spokes=64, spoke_length=64,
                 embed_dim=768, 
                 use_real_imaginary=True):
        super().__init__()
        self.num_spokes = num_spokes
        self.spoke_length = spoke_length
        self.embed_dim = embed_dim

        # will combine both channels like in original ViT
        print("Combining two channels before linear projection")
       
        self.proj = nn.Linear(2*spoke_length, embed_dim)  # combining two channels

        # NOTE: post norm?? I guess not needed as both outputs are by nn linear and already normalized
        # Also, attn blocks have their own pre norms
      

    def forward(self, x):
        # x is radial kspaace torch tensors [bs, num_spokes, spoke_length] -> it is a complex tensor
        ch1 = torch.real(x) # real spectrum
        ch2 = torch.imag(x) # imag spectrum
        
        # NOTE: no need now, separate channels are not complex
        # lets first convert to complex64 because learnable matrix are float not double
        ch1 = ch1.to(torch.float32) # converting from double to float
        ch2 = ch2.to(torch.float32)

        B, num_spokes, spoke_length = ch1.shape # [bs, num_spokes, spoke_length]
        
        # FIXME look at relaxing size constraints
        assert num_spokes == self.num_spokes and spoke_length == self.spoke_length, \
            f"Input kspace size ({num_spokes}*{spoke_length}) doesn't match model ({self.num_spokes}*{self.spoke_length})."
        
        # passing thru linear layer
        ch1 = rearrange(ch1, "bs num_spokes spoke_length -> (bs num_spokes) spoke_length")
        ch2 = rearrange(ch2, "bs num_spokes spoke_length -> (bs num_spokes) spoke_length")
        
        concat_channel = torch.cat([ch1, ch2], dim=-1)
        assert concat_channel.shape[-1] == 2*self.spoke_length, \
            f"concat_channel dim ({concat_channel.shape[-1]}) doesn't match model 2*spoke_length ({2*self.spoke_length})."

        embedding = self.proj(concat_channel) # shape [(bs num_spokes) embed_dim]

        embedding = rearrange(embedding, "(bs sq_len) embed_dim -> bs sq_len embed_dim", bs=B, sq_len=num_spokes)

        return embedding


class RadialKspaceCombinedTransformer(nn.Module):
    """ Radial Kspace Transformer
    """
    def __init__(self, num_spokes=64, spoke_length=64, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 use_real_imaginary=True,
                 spokes_selection_rate=0.25,
                 learned_pos_encoding=False,
                 global_pool=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
        self.token_embed = RadialKspaceEmbedCombinedLinear(num_spokes=num_spokes, spoke_length=spoke_length, embed_dim=embed_dim, use_real_imaginary=use_real_imaginary)
        num_tokens = num_spokes 

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if learned_pos_encoding:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim), requires_grad=False)
            pos_embed = positionalencoding1d(embed_dim, num_tokens + 1)
            self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))


        self.pos_drop = nn.Dropout(p=drop_rate)

        self.spokes_selection_rate = spokes_selection_rate

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.global_pool = global_pool
        print("Using global pool instead of CLS token")

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_hook=False):
        # x is radial kspaace torch tensors [bs, num_spokes, spoke_length] -> it is a complex tensor
        B = x.shape[0]
        num_spokes = x.shape[1]
        x = self.token_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        

        # x.shape: [bs, num_spokes+1, emb_dim]
        if self.spokes_selection_rate < 1.0:
            # select only a limited spokes for input to the transformer
            number_of_spokes_to_select = int(self.spokes_selection_rate*num_spokes)
            cls_tokens = x[:, 0:1, :] 

            randomly_selected_indices = torch.stack([torch.randperm(num_spokes)[:number_of_spokes_to_select] for _ in range(B)]) # shape [b, m]
            batch_indices = torch.arange(B).unsqueeze(1)  # shape [b, 1]

            randomly_selected_tokens = x[batch_indices, randomly_selected_indices + 1] # shape [b, m, d]
            x = torch.cat((cls_tokens, randomly_selected_tokens), dim=1) # should be of shape [bs, 1 + selected tokens, emb_dim]
        else:
            x = self.pos_drop(x)

        # print("x.shape just before transformer blocks: ", x.shape) # [bs, num_spokes+1, emb_dim]

        for blk in self.blocks:
            x = blk(x, register_hook=register_hook)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0] # or just take the cls token

        x = self.head(x)
        return x

