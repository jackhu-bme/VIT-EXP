import math
import copy
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from ct_clip.mlm import MLM
from ct_clip.visual_ssl import SimSiam, SimCLR
from ct_clip.distributed import AllGather

from transformers import BertTokenizer, BertModel

import time

import random

import matplotlib.pyplot as plt

from ct_clip.utils import vis_3d_img_list

import wandb

from sklearn.metrics import roc_auc_score

import numpy as np

import nibabel as nib

import os

# helper functions

def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def calculate_vis_auc(input, target):
    """
    Calculate AUC and log the ROC curve using WandB.

    Parameters:
    - input: numpy array or torch tensor of predicted probabilities (0-1 range).
    - target: numpy array or torch tensor of binary ground truth values (0 or 1).
    """
    # Convert to numpy if input and target are tensors
    if not isinstance(input, np.ndarray):
        input = input.cpu().detach().numpy()
    if not isinstance(target, np.ndarray):
        target = target.cpu().detach().numpy()
    
    # Flatten the inputs and targets
    input_flat = input.flatten()
    target_flat = target.flatten()

    target_flat[target_flat > 0.5] = 1
    target_flat[target_flat <= 0.5] = 0
    # to int
    target_flat = target_flat.astype(int)

    # print(f"input flat shape: {input_flat.shape}, type: {type(input_flat)}")
    # print(f"target flat shape: {target_flat.shape}, type: {type(target_flat)}")
    
    # Calculate AUC
    try:
        auc_score = roc_auc_score(target_flat.copy(), input_flat.copy())
        print(f"AUC Score: {auc_score}")

        pred = np.concatenate((1-input_flat.reshape(-1,1), input_flat.reshape(-1,1)), axis=1)
    except Exception as e:
        print(f"skipping auc calculation due to error: {e}, possibly due to all 0 or 1 ground truth")
        return None, None
    
    return auc_score, wandb.plot.roc_curve(target_flat, pred, ["Negative", "Positive"])


# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w z) c -> b c h w z', h = h_r, w= w_r)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x, force_keep_all = False):
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# rotary positional embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# transformer

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, causal = False, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias = False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, rotary_pos_emb = None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            dim_head = 64,
            heads = 8,
            causal = False,
            attn_dropout = 0.,
            ff_dropout = 0.,
            ff_mult = 4,
            checkpoint_during_training = False
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult)),
            ]))

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(
            self,
            x,
            rotary_pos_emb = None,
            mask = None
    ):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
            x = ff(x) + x

        return self.norm_out(x)

# text and vision transformers

class TextTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            num_tokens,
            max_seq_len,
            dim_head,
            rotary_pos_emb = None,
            causal = False,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None

        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        self.transformer = Transformer(dim, dim_head = dim_head, causal = causal, **kwargs)

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1, device = device)

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        out = self.transformer(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
        return out

class VisionTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            image_size,
            patch_size,
            channels,
            patch_dropout = 0.5,
            **kwargs
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_emb = nn.Embedding(num_patches, dim)
        self.patch_dropout = PatchDropout(patch_dropout)

        self.transformer = Transformer(dim, **kwargs)

        self.to_cls_tokens = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim, bias = False),
            Rearrange('b d -> b 1 d')
        )

    def forward(
            self,
            x,
            keep_all_patches = False
    ):
        device = x.device

        x = self.to_tokens(x)
        b, n, _ = x.shape

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        x = self.patch_dropout(x, force_keep_all = keep_all_patches)

        out = self.transformer(x)

        cls_tokens = self.to_cls_tokens(out)
        return torch.cat((cls_tokens, out), dim = 1)

# contrastive learning functions

def model_forward_with_context(
        *,
        fn,
        args,
        freeze,
):
    encoding_context = null_context if not freeze else torch.no_grad

    with encoding_context():
        enc = fn(*args)

        if freeze:
            enc.detach_()

    return enc

# main clip class

class CTCLIP(nn.Module):
    def __init__(
            self,
            *,
            image_encoder = None,
            text_encoder = None,
            dim_text = 512,
            dim_image = 512,
            dim_latent = 512,
            num_text_tokens = 28897,
            text_enc_depth = 6,
            text_seq_len = 256,
            text_heads = 8,
            text_dim_head = 64,
            text_has_cls_token = False,
            text_pad_id = 0,
            text_rotary_pos_emb = False,
            text_causal_mask = False,
            text_eos_id = None,
            text_encode_without_mask = False,
            visual_enc_depth = 6,
            visual_heads = 8,
            visual_dim_head = 64,
            visual_image_size = 256,
            visual_patch_size = 32,
            visual_patch_dropout = 0.5,
            visual_has_cls_token = False,
            channels = 3,
            use_all_token_embeds = False,
            downsample_image_embeds = False,
            decoupled_contrastive_learning = False,
            extra_latent_projection = False,
            use_mlm = False,
            text_ssl_loss_weight = 0.05,
            use_visual_ssl = False,
            visual_ssl = None,
            visual_ssl_type = 'simsiam',
            visual_ssl_hidden_layer = -1,
            simclr_temperature = 0.1,
            image_ssl_loss_weight = 0.05,
            multiview_loss_weight = 0.1,
            checkpoint_during_training = False,
            config = None,
            **kwargs
    ):
        super().__init__()
        assert use_all_token_embeds == False, "no more support for use_all_token_embeds"
        assert extra_latent_projection == False, "no more support for extra_latent_projection"
        #assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'
        self.dtype=torch.float32
        # store some parameters for access

        self.config = config

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        # instantiate text transformer

        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len

        self.text_encode_without_mask = text_encode_without_mask # whether to pass in text mask to text encoder

        self.text_causal_mask = text_causal_mask
        self.text_eos_id = text_eos_id

        assert not (text_causal_mask and not exists(text_eos_id)), 'text EOS token id must be given if using causal mask in text transformer'

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = TextTransformer(
                dim = dim_text,
                num_tokens = num_text_tokens + (1 if use_mlm else 0),
                max_seq_len = text_seq_len,
                depth = text_enc_depth,
                heads = text_heads,
                causal = text_causal_mask,
                dim_head = text_dim_head,
                rotary_pos_emb = text_rotary_pos_emb,
                checkpoint_during_training = checkpoint_during_training
            )

        # instantiate image transformer

        self.visual_has_cls_token = visual_has_cls_token

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = VisionTransformer(
                dim = dim_image,
                image_size = visual_image_size,
                patch_size = visual_patch_size,
                channels = channels,
                depth = visual_enc_depth,
                heads = visual_heads,
                dim_head = visual_dim_head,
                patch_dropout = visual_patch_dropout,
                checkpoint_during_training = checkpoint_during_training
            )

        # text ssl

        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        if use_mlm:
            mlm_kwargs, kwargs = groupby_prefix_and_trim('mlm_', kwargs)
            self.mlm = MLM(
                self.text_transformer,
                dim = dim_text,
                num_tokens = num_text_tokens,
                **mlm_kwargs
            )

        # image ssl

        self.use_visual_ssl = use_visual_ssl or exists(visual_ssl)
        self.image_ssl_loss_weight = image_ssl_loss_weight if use_visual_ssl else 0

        if self.use_visual_ssl:
            if exists(visual_ssl):
                self.visual_ssl = visual_ssl

            elif use_visual_ssl:
                if visual_ssl_type == 'simsiam':
                    ssl_type = partial(SimSiam, channels = channels)
                elif visual_ssl_type == 'simclr':
                    ssl_type = partial(SimCLR, temperature = simclr_temperature, channels = channels)
                else:
                    raise ValueError(f'unknown visual_ssl_type')

                self.visual_ssl = ssl_type(
                    self.visual_transformer,
                    image_size = visual_image_size,
                    hidden_layer = visual_ssl_hidden_layer
                )

        # text latent projection

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        # image latent projection

        if downsample_image_embeds:
            #assert use_all_token_embeds, 'must be using all token embeds for contrastive learning in order to downsampling'
            dim_conv=512
            self.to_visual_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv3d(dim_conv, dim_conv, 4, stride = 2, padding = 1, bias = False, groups = dim_conv),
                nn.Conv3d(dim_conv, dim_latent, 1),
                Rearrange('b c h w z -> b (h w z c)'),
                nn.Linear(dim_image, dim_latent, bias = False)
            )
        else:
            self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        # temperature

        self.temperature = nn.Parameter(torch.tensor(1.))

        # from https://arxiv.org/abs/2111.07783 (FILIP paper)
        self.use_all_token_embeds = use_all_token_embeds

        # proposed in https://arxiv.org/abs/2110.06848 (DCL) and https://arxiv.org/abs/2110.11316 (CLOOB)
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        # proposed in https://arxiv.org/abs/2110.11316 (CLOOB)
        self.extra_latent_projection = extra_latent_projection

        self.to_text_latent_extra = copy.deepcopy(self.to_text_latent)

        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)

        self.multiview_loss_weight = multiview_loss_weight

        self.tokenizer= BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

        self.seg_criterion = nn.BCEWithLogitsLoss()

        self.fix_text_encoder = config.get("fix_text_encoder", False)

        self.use_seg = config.get("use_seg", False)
        if self.use_seg:
            seg_head_config = config.get("seg_head", {})
            out_final_dim = seg_head_config.get("head_out_dim", 22) * self.visual_transformer.patch_voxel_nums
            seg_head_kwargs = dict(
                head_n_layers = seg_head_config.get("head_n_layers", 2),
                head_layer_type = seg_head_config.get("head_layer_type", "mlp"),
                head_in_dim = seg_head_config.get("head_in_dim", 256),
                head_mid_dim = seg_head_config.get("head_mid_dim", 128),
                head_out_dim = out_final_dim,
            )
            self.seg_head = self.create_head(**seg_head_kwargs)
        else:
            self.seg_head = None

        self.use_open_seg = config.get("use_open_seg", False)
        if self.use_open_seg:
            # open seg head
            open_seg_head_config = config.get("open_seg_head", {})
            open_seg_head_kwargs = dict(
                head_n_layers = open_seg_head_config.get("n_layers", 2),
                head_layer_type = open_seg_head_config.get("layer_type", "mlp"),
                head_in_dim = open_seg_head_config.get("in_dim", 256),
                head_mid_dim = open_seg_head_config.get("mid_dim", 128),
                head_out_dim = open_seg_head_config.get("out_dim", 16) * self.visual_transformer.patch_voxel_nums
            )
            self.open_seg_head = self.create_head(**open_seg_head_kwargs)
            # open text head
            open_text_head_config = config.get("open_text_head", {})
            open_text_head_kwargs = dict(
                head_n_layers = open_text_head_config.get("n_layers", 2),
                head_layer_type = open_text_head_config.get("layer_type", "mlp"),
                head_in_dim = open_text_head_config.get("in_dim", 768),
                head_mid_dim = open_text_head_config.get("mid_dim", 128),
                head_out_dim = open_text_head_config.get("out_dim", 16)
            )
            self.open_text_head = self.create_head(**open_text_head_kwargs)
            self.open_seg_loss_type = config.get("open_seg_loss_type", "cos_sim_l2")
            self.open_seg_loss_down_factor = int(config.get("open_seg_loss_down_factor", 1))
            self.open_seg_loss_hyper_config = config.get("open_seg_loss_hyper_config", {})
            self.bce_criterion = nn.BCELoss()
            self.bce_no_reduction_criterion = nn.BCELoss(reduction='none')
        else:
            self.open_seg_head = None
            self.open_text_head = None
            self.open_seg_loss_type = None
            self.open_seg_loss_down_factor = None
       

    @staticmethod
    def create_head(head_n_layers, head_layer_type, head_in_dim, head_mid_dim, head_out_dim):
        # warning: todo: check output logits, if it is consistent with loss design!
        if head_layer_type == "mlp":
            layers = []
            for i in range(head_n_layers):
                in_dim = head_in_dim if i == 0 else head_mid_dim
                out_dim = head_out_dim if i == head_n_layers - 1 else head_mid_dim
                act = nn.LeakyReLU(0.2) if i < head_n_layers - 1 else nn.Identity()
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    act
                ])
            return nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported head_layer_type: {head_layer_type}")



    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path, check=True):
        path = Path(path)
        assert path.exists()
        # try:
        pt = torch.load(str(path), map_location='cpu')
        #     self.load_state_dict(pt)
        # except Exception as e:
        #     if not check:
        #         raise e
        #     else:
        # print(f'failed to load model dirctly from {path}, due to error: {e}, try remove the module name from state dict')
        pt_new = {k[7:]:v for k,v in pt.items()}
        self.load_state_dict(pt_new)
        del pt, pt_new
        print(f'successfully loaded model from {path}')


    def tokenize(self, prompt):
        text_tokens=self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(torch.cuda)
        return text_tokens
    
    def token_embedding(self,input_ids):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if hasattr(self.text_transformer.embeddings, "token_type_ids"):
            print("hahatrue")

        buffered_token_type_ids = self.text_transformer.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded
        text_embeddings = self.text_transformer.embeddings(input_ids = input_ids, token_type_ids = token_type_ids)
        return text_embeddings


    def forward_infer(self, text, image, buffer_text_embed = None, buffer_image_embed = None):
        # start_time = time.time()
        if buffer_text_embed is None:
            # derive text mask
            text_mask =text.attention_mask
            # get encoded text
            text_args = (text.input_ids,text.attention_mask)
            if not self.text_encode_without_mask:
                text_args = (*text_args, text_mask)
            text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
        else:
            # print("buffer text embed")
            text_embeddings = buffer_text_embed

        enc_text = text_embeddings[0]

        if buffer_image_embed is None:
            enc_image= self.visual_transformer(image, return_encoded_tokens=True)
        else:
            # print("buffer image embed shape: ", buffer_image_embed.shape)
            enc_image = buffer_image_embed
        # print(f"encoded image shape: {enc_image.shape}")
        #print("This is visual encoding")

        # step_1_time = time.time()-start_time
        # print(f"Time taken for step 1: {step_1_time}")

        global h_r, w_r, z_r
        h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]

        #enc_image, max_indices = torch.max(enc_image, dim=1)

        # step_2_time = time.time()-start_time-step_1_time
        # print(f"Time taken for step 2: {step_2_time}")

        enc_image = torch.mean(enc_image, dim=1)
        enc_image = enc_image.view(enc_image.shape[0], -1)

        # step_3_time = time.time()-start_time-step_1_time-step_2_time
        # print(f"Time taken for step 3: {step_3_time}")

        # early return of encodings, if needed (for DALL-E2)
        text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
        image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image

        # step_4_time = time.time()-start_time-step_1_time-step_2_time-step_3_time
        # print(f"Time taken for step 4: {step_4_time}")

        # project to latents
        #text_embeds = text_embeds.view(text_embeds.shape[0], -1)
        text_embeds = text_embeds[:,0,:]


        text_latents = self.to_text_latent(text_embeds)

        image_latents = self.to_visual_latent(image_embeds)
        text_latents, image_latents = map(l2norm, (text_latents, image_latents))

        # print(f"in forward infer, shape of text_latents: {text_latents.shape} and image_latents: {image_latents.shape}")

        # broadcast the text latents to match with the image latents if shape mismatch

        # if text_latents.shape[0] != image_latents.shape[0]:
        #     print(f"warning: text_latents shape: {text_latents.shape} and image_latents shape: {image_latents.shape} mismatch, broadcasting text_latents to match image_latents")
        #     text_latents = text_latents.expand_as(image_latents)

        temp = self.temperature.exp()

        einsum_args = text_latents, image_latents
        res = einsum('b d, b d -> b', *einsum_args) * temp
        return res

    def forward(self, batch, device=None, accelerator=None, **kwargs):
        # define the forward (data to loss) logic for different types of data
        # in the ref version, in a single batch, only one type of data is present
        if batch["data_type"][0] == "imagereport":
            return self.forward_batch_image_report(batch, device=device, accelerator=accelerator, **kwargs)
        elif batch["data_type"][0] == "imageseg":
            return self.forward_batch_image_seg(batch, device=device, accelerator=accelerator, **kwargs)
        elif batch["data_type"][0] == "imageopenseg":
            return self.forward_batch_image_open_seg(batch, device=device, accelerator=accelerator, **kwargs)
        else:
            raise ValueError(f"Data type {batch['data_type']} not recognized")
    
    def open_seg_loss(self, seg_preds, seg_mask_flatten, prompt_logits_batch):
        # seg_preds: [B, L, n_hidden_dim=16]
        # seg_mask_flatten: [B, L, C]
        # prompt_logits_batch: [B, C, n_hidden_dim=16]
        # print(f"seg_preds shape: {seg_preds.shape}")
        # print(f"seg_mask_flatten shape: {seg_mask_flatten.shape}")
        # print(f"prompt_logits_batch shape: {prompt_logits_batch.shape}")
        # exit()
        if self.open_seg_loss_type == "cos_sim_l2":
            # calculate the cosine similarity for each class
            B, L, n_hidden_dim = seg_preds.shape
            B, L, C = seg_mask_flatten.shape
            open_seg_loss = 0.
            # continue_train = input("Continue training? 3")
            for i in range(C):
                # get the prompt logits for the i-th class
                prompt_logits = prompt_logits_batch[:, i, :] # [B, n_hidden_dim=16]
                # continue_train = input("Continue training? 4")
                sim = (F.cosine_similarity(seg_preds, prompt_logits.unsqueeze(1), dim=-1) + 1) / 2
                # print(f"sim shape: {sim.shape}") # (B, L)
                # calculate the distance between similarity and gt, make the right class close to 1, wrong class close to 0
                # just use l2 loss for now
                # continue_train = input("Continue training? 5")
                open_seg_loss += F.mse_loss(sim, seg_mask_flatten[:, :, i]) # default reduction is mean
                # continue_train = input("Continue training? 6")
                # empty the memory
                torch.cuda.empty_cache()
            print(f"open_seg_loss shape: {open_seg_loss.shape}")
            return open_seg_loss
        elif self.open_seg_loss_type == "clip_loss":
            temp = self.open_seg_loss_hyper_config.get("temp", 0.1)
            # calculate the cosine similarity for each class
            B, L, C = seg_mask_flatten.shape
            open_seg_loss = 0.
            # continue_train = input("Continue training? 3")
            # sim_all = F.cosine_similarity(seg_preds.unsqueeze(2), prompt_logits_batch.unsqueeze(1), dim=-1) # [B, L, C]
            sim_logits = torch.einsum('bld,bcd->blc', seg_preds, prompt_logits_batch) / temp # [B, L, C]
            # now i would like to use the similarity to contrast with [0, 1, 0, 0, 1, .., 0, 1] multi-hot vector gts for each class
            open_seg_loss = F.cross_entropy(sim_logits.reshape(-1, C), seg_mask_flatten.reshape(-1, C)) # / B
            return open_seg_loss
        elif self.open_seg_loss_type == "clip_bce_loss":
            # calculate the cosine similarity for each class
            B, L, C = seg_mask_flatten.shape
            open_seg_loss = 0.
            sim_all = (F.cosine_similarity(seg_preds.unsqueeze(2), prompt_logits_batch.unsqueeze(1), dim=-1) + 1) / 2 # [B, L, C]
            # sim_logits = torch.einsum('bld,bcd->blc', seg_preds, prompt_logits_batch)
            open_seg_loss = self.bce_criterion(sim_all.reshape(-1, C), seg_mask_flatten.reshape(-1, C))
            return open_seg_loss 
        elif self.open_seg_loss_type == "clip_focal_loss":
            gamma = self.open_seg_loss_hyper_config.get("gamma", 2)
            alpha = self.open_seg_loss_hyper_config.get("alpha", 0.25)
            # calculate the cosine similarity for each class
            B, L, C = seg_mask_flatten.shape
            open_seg_loss = 0.
            # continue_train = input("Continue training? 3")
            sim_all = (F.cosine_similarity(seg_preds.unsqueeze(2), prompt_logits_batch.unsqueeze(1), dim=-1) + 1) / 2 # [B, L, C]
            p = sim_all.reshape(-1, C)
            targets = seg_mask_flatten.reshape(-1, C)
            # bce_loss = self.bce_criterion(p, targets, reduction='none')
            bce_loss = self.bce_no_reduction_criterion(p, targets)

            p_t = p * targets + (1 - p) * (1 - targets)
    
            loss = bce_loss * ((1 - p_t) ** gamma)
            if alpha >= 0:
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                loss = alpha_t * loss
            open_seg_loss = loss.mean()
            return open_seg_loss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        else:
            raise ValueError(f"Unsupported open seg loss type: {self.open_seg_loss_type}")

    @staticmethod
    def random_downsample(tensor, factor, start_index=None):
        # use index to downsample the tensor
        assert tensor.ndim == 5 and isinstance(factor, int)
        # B, C, H, W, D = tensor.shape
        # downsample the tensor
        if start_index is None:
            start_index = random.randint(0, factor-1)
        new_tensor = tensor[:, :, ::factor, ::factor, ::factor]
        return new_tensor, start_index



    def forward_batch_image_open_seg(self, batch, device=None, accelerator=None, **kwargs):
        # start_time = time.time()
        image = batch["image"]
        seg_mask = batch["seg_mask"] # [B, C, H, W, D]
        B_seg, C_seg, D, W, H = seg_mask.shape
        seg_mask, start_index = self.random_downsample(seg_mask, self.open_seg_loss_down_factor, start_index=None)
        seg_mask_flatten = seg_mask.permute((0, 2, 3, 4, 1)).reshape(B_seg, -1, C_seg) # (B, L, C)
        seg_mask_promp_dict = batch["seg_mask_promp_dict"]
        seg_mask_prompt_list = list(seg_mask_promp_dict.values()) # [C tensors of shape (B, length=512, )], just input_ids for bert model!
        seg_mask_prompt_list = [prompt[0:1] for prompt in seg_mask_prompt_list] # as all the samples use same classes, just take the first one sample
        seg_mask_prompts = torch.cat(seg_mask_prompt_list, dim=0) # already tokens, [C, length=512], C=num_labels
        # print(f"seg_mask_prompts shape: {seg_mask_prompts.shape}")
        # get text embeddings by text transformers
        if self.fix_text_encoder:
            self.text_transformer.eval()
        seg_prompt_text_embeddings = self.text_transformer(seg_mask_prompts)[0] # [C, length=512, n_hidden_dim]
        # step_1_time = time.time()-start_time
        # print(f"Time taken for step 1: {step_1_time}")
        # print(f"seg_prompt_text_embeddings shape: {seg_prompt_text_embeddings.shape}")
        # exit()
        
        # follow the way in ct-clip to get the latents: the first token of the text embeddings
        seg_prompt_latents = seg_prompt_text_embeddings[:, 0, :] # [C, n_hidden_dim]

        # get a lower dimension embedding with mlp
        prompt_logits = self.open_text_head(seg_prompt_latents).unsqueeze(0) # [1, C, 16]
        # print(f"prompt_logits shape: {prompt_logits.shape}")
        # exit()
        prompt_logits_batch = torch.tile(prompt_logits, (B_seg, 1, 1)) # [B, C, n_hidden_dim=16]
        low_latent_dim = prompt_logits_batch.shape[-1]
        # print(f"prompt_logits_batch shape: {prompt_logits_batch.shape}")
        loss_dict = {}
        B, C, D, W, H = image.shape

        # step_2_time = time.time()-start_time-step_1_time
        # print(f"Time taken for step 2: {step_2_time}")


        enc_image= self.visual_transformer(image, return_encoded_tokens=True)

        # step_3_time = time.time()-start_time-step_1_time-step_2_time
        # print(f"Time taken for step 3: {step_3_time}")
        # continue_train = input("Continue training? 1")
        # use the seg valid mask to choose the image to be segmented
        # due to memory issues, use one for seg only now
        seg_mask = seg_mask.float().to(device)
        enc_seg_image = enc_image
        b, d, w, h, c = enc_seg_image.shape
        p_h, p_w, p_d = H//h, W//w, D//d
        tokens_to_seg = enc_seg_image.reshape(-1, c) # b, l, c -> b*l, c
        # print(f"tokens_to_seg shape: {tokens_to_seg.shape}")
        # use the linear head for 
        seg_logits = self.open_seg_head(tokens_to_seg)

        # step_4_time = time.time()-start_time-step_1_time-step_2_time-step_3_time
        # print(f"Time taken for step 4: {step_4_time}")
        # continue_train = input("Continue training? 2")
        # reshape the logits to the original shape, with each pixel
        seg_preds = seg_logits.view(b, d, w, h, p_d, p_w, p_h, -1)
        seg_preds = seg_preds.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, -1, D, W, H) # B, C, D, W, H as voxel embeddings

        # step_5_time = time.time()-start_time-step_1_time-step_2_time-step_3_time-step_4_time
        # print(f"Time taken for step 5: {step_5_time}")
        # downsample the seg_logits
        seg_preds = self.random_downsample(seg_preds, self.open_seg_loss_down_factor, start_index=start_index)[0]
        seg_preds = seg_preds.permute((0, 2, 3, 4, 1)).reshape(B_seg, -1, low_latent_dim) # (B, L, n_hidden_dim=16)

        # step_6_time = time.time()-start_time-step_1_time-step_2_time-step_3_time-step_4_time-step_5_time
        # print(f"Time taken for step 6: {step_6_time}")
        open_seg_loss = self.open_seg_loss(seg_preds, seg_mask_flatten, prompt_logits_batch) # keep the start index same
        # step_7_time = time.time()-start_time-step_1_time-step_2_time-step_3_time-step_4_time-step_5_time-step_6_time
        # print(f"Time taken for step 7: {step_7_time}")
        # exit()
        loss_dict["open_seg_loss"] = open_seg_loss.item()
        return_list = [open_seg_loss, loss_dict]

        # visualize when training, just the things to be logged in wandb
        vis = kwargs.get("return_visualize", False) or kwargs.get("return_vis", False)
        down_img = self.random_downsample(image, self.open_seg_loss_down_factor, start_index=start_index)[0]
        B, C, D_down, W_down, H_down = down_img.shape
        if vis:
            img_prefix = kwargs.get("img_prefix", "")
            # visualize the normed similarity and the gt mask for each class
            with torch.no_grad():
                vis_dict = {}
                # visualize the segmentation results, for each image in the batch, with slices plot
                local_save = False
                if local_save:
                    local_vis_save_dir = "./wandb_nii_gz"
                    os.makedirs(local_vis_save_dir, exist_ok=True)
                for i in range(C_seg):
                    # get the prompt logits for the i-th class
                    prompt_logits = prompt_logits_batch[:, i, :] # [B, n_hidden_dim=16]
                    # continue_train = input("Continue training? 4")
                    sim = F.cosine_similarity(seg_preds, prompt_logits.unsqueeze(1), dim=-1)
                    sim = (sim + 1) / 2  # fix the range to [0, 1], after v3-5 exps!
                    sim_vis_0 = sim.reshape(B_seg, D_down, W_down, H_down)[0]
                    mask_gt_vis_0 = seg_mask_flatten[:, :, i].reshape(B_seg, D_down, W_down, H_down)[0]
                    down_img_vis_0 = down_img[0, 0].reshape(D_down, W_down, H_down)
                    # vis the similarity, gt mask, and downsampled image
                    img_name = f"{img_prefix}_channel_{i}_seg" if img_prefix else f"channel_{i}"
                    vis_res = vis_3d_img_list([down_img_vis_0, sim_vis_0, mask_gt_vis_0], img_name=img_name)
                    # save image at local, for debug vis only
                    if local_save:
                        down_img_arr = down_img_vis_0.cpu().numpy() * 1000
                        down_img_nii = nib.Nifti1Image(down_img_arr, np.eye(4))
                        nib.save(down_img_nii, os.path.join(local_vis_save_dir, f"{img_name}_down_img_channel_{i}.nii.gz"))
                        sim_img = sim_vis_0.cpu().numpy() * 1000
                        sim_img_nii = nib.Nifti1Image(sim_img, np.eye(4))
                        nib.save(sim_img_nii, os.path.join(local_vis_save_dir, f"{img_name}_sim_channel_{i}.nii.gz"))
                        mask_img = mask_gt_vis_0.cpu().numpy() * 1000
                        mask_img_nii = nib.Nifti1Image(mask_img, np.eye(4))
                        nib.save(mask_img_nii, os.path.join(local_vis_save_dir, f"{img_name}_mask_channel_{i}.nii.gz"))
                    # update the vis dict with each key-value pair
                    vis_dict.update(vis_res)
                # visulize the auc based on the cosine simliarity and whether this voxel is positive or negative for each class
                # todo: fix the problem of ploting auc in the future (np.float64 unsupport type!), tmp just do not cal and log it
                # for i in range(C_seg):
                #     # get the prompt logits for the i-th class
                #     prompt_logits = prompt_logits_batch[:, i, :]
                #     sim = F.cosine_similarity(seg_preds, prompt_logits.unsqueeze(1), dim=-1)
                #     sim = (sim + 1) / 2  # fix the range to [0, 1], after v3-5 exps!
                #     sim_vis_0 = sim.reshape(B_seg, D_down, W_down, H_down)[0]
                #     mask_gt_vis_0 = seg_mask_flatten[:, :, i].reshape(B_seg, D_down, W_down, H_down)[0]
                #     auc, auc_plot = calculate_vis_auc(sim_vis_0, mask_gt_vis_0)
                #     vis_dict[f"auc_channel_{i}"] = auc
                #     vis_dict[f"auc_plot_channel_{i}"] = auc_plot
                return_list.append(vis_dict)
                # exit()
        return return_list

    
    def forward_batch_image_seg(self, batch, device=None, accelerator=None, return_metrics=False, return_vis=False, **kwargs):
        image = batch["image"]
        seg_mask = batch["seg_mask"]
        loss_dict = {}
        B, C, D, W, H = image.shape
        enc_image= self.visual_transformer(image, return_encoded_tokens=True)
        # use the seg valid mask to choose the image to be segmented
        # due to memory issues, use one for seg only now
        seg_mask = seg_mask.float().to(device)
        enc_seg_image = enc_image
        b, d, w, h, c = enc_seg_image.shape
        p_h, p_w, p_d = H//h, W//w, D//d
        tokens_to_seg = enc_seg_image.reshape(-1, c) # b, l, c -> b*l, c
        # use the linear head for 
        seg_logits = self.seg_head(tokens_to_seg)
        # reshape the logits to the original shape, with each pixel
        seg_preds = seg_logits.reshape(b, d, w, h, p_d, p_w, p_h, -1)
        seg_preds = seg_preds.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, -1, D, W, H)
        # seg_mask = seg_mask.float().to(device)
        seg_loss = self.seg_criterion(seg_preds, seg_mask) # B, C, D, W, H
        loss_dict['seg_loss'] = seg_loss.item()
        return_list = [seg_loss, loss_dict]
        if return_metrics:
            metrics_dict = {}
            # calculate the dice score for each channel
            with torch.no_grad():
                seg_preds = torch.sigmoid(seg_preds)
                seg_preds = (seg_preds > 0.5).float()
                intersection = torch.sum(seg_preds * seg_mask, dim=(2,3,4))
                union = torch.sum(seg_preds, dim=(2,3,4)) + torch.sum(seg_mask, dim=(2,3,4))
                dice_scores = 2*intersection / union
                dice_scores = dice_scores.mean(dim=0)
                metrics_dict["dice_score"] = dice_scores.cpu().numpy()
            return_list.append(metrics_dict)
        if return_vis:
            # visualize the segmentation results, for each image in the batch
            vis_dict = {}
            # vis the original image, mask, and preds, seperate for each channel
            # choose slices from each view, [0.2, 0.4, 0.6, 0.8] ratios of each axis
            # get a vis map for each image, which could be output to png directly
            with torch.no_grad():
                seg_preds = torch.sigmoid(seg_preds)
                # seg_preds = (seg_preds > 0.5).float()
                # sample on each axis
                ratio_list = [0.25, 0.5, 0.75]
                axial_indices = [int(D*ratio) for ratio in ratio_list]
                coronal_indices = [int(W*ratio) for ratio in ratio_list]
                sagittal_indices = [int(H*ratio) for ratio in ratio_list]
                # get the slice for each axis
                _, C_seg, _, _, _ = seg_mask.shape
                image = torch.repeat_interleave(image, C_seg, 1)
                all_results = torch.stack([image, seg_mask, seg_preds], dim=-1) # (B, C, D, W, H, 3) 3 for 3 types of images
                axial_slices = torch.stack([all_results[0, :, idx, :, :, :] for idx in axial_indices], dim=-1) # (C, W, H, 3, 3) 3 in ratio list
                coronal_slices = torch.stack([all_results[0, :, :, idx, :, :] for idx in coronal_indices], dim=-1) # (C, D, H, 3, 3) 3 in ratio list
                sagittal_slices = torch.stack([all_results[0, :, :, :, idx, :] for idx in sagittal_indices], dim=-1) # (C, D, W, 3, 3) 3 in ratio list
                vis_dict["axial_slices"] = axial_slices.cpu()  # .numpy()
                vis_dict["coronal_slices"] = coronal_slices.cpu()  # .numpy()
                vis_dict["sagittal_slices"] = sagittal_slices.cpu()  # .numpy()
            return_list.append(vis_dict)
        return return_list


    def forward_batch_image_report(self, batch, device=None, accelerator=None, **kwargs):
        text = batch["text"] # attention here, this is after the tokenization, text tokens actually! follow the ctclip code temporally
        image = batch["image"]
        # print(f"loaded text: {text}")
        b, device = text.input_ids.shape[0], device
        # derive text mask
        text_mask = text.attention_mask
        loss_dict = {}
        num_batch_texts = num_batch_images = 1
        is_multiview = False

        # get encoded text
        text_args = (text.input_ids,text.attention_mask)

        if not self.text_encode_without_mask:
            text_args = (*text_args, text_mask)

        if self.fix_text_encoder:
            self.text_transformer.eval()
        text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask)
        enc_text = text_embeddings[0]

        enc_image= self.visual_transformer(image, return_encoded_tokens=True) # (B, H, W, Z, C)

        print(f"encoded image shape: {enc_image.shape}")
        # exit()
        #print("This is visual encoding")
        global h_r, w_r, z_r
        (B, h_r, w_r, z_r, C_img) = enc_image.shape
        #enc_image, max_indices = torch.max(enc_image, dim=1)
        # enc_image_send = enc_image

        # enc_image = torch.mean(enc_image, dim=1)
        # enc_image = enc_image.view(enc_image.shape[0], -1)
        enc_image = enc_image.view(-1, C_img)

        image_latents_all = self.to_visual_latent(image_embeds)

        image_latents_all = image_latents_all.reshape(B, h_r * w_r * z_r, -1)

        print(f"image latents all shape: {image_latents_all.shape}")

        # mean on dim 1 to get the image latents
        image_latents = torch.mean(image_latents_all, dim=1)

        print(f"image latents shape: {image_latents.shape}")
        exit()

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only

        text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
        image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image

        # project to latents
        #text_embeds = text_embeds.view(text_embeds.shape[0], -1)
        text_embeds = text_embeds[:,0,:]

        
        #text_embeds = torch.mean(text_embeds, dim=1)
        text_latents = self.to_text_latent(text_embeds)


        text_latents, image_latents = map(l2norm, (text_latents, image_latents))

        # get temperature

        temp = self.temperature.exp()

        # split out multiview dimension for text and images

        bs_single_gpu = text_latents.shape[0]

        # gather
        assert accelerator is not None, "accelerator is not provided"

        text_latents_gather = AllGather.apply(text_latents, accelerator)
        image_latents_gather = AllGather.apply(image_latents, accelerator)

        text_latents_gather = rearrange(text_latents_gather, '(m b) ... -> m b ...', m = num_batch_texts)
        image_latents_gather = rearrange(image_latents_gather, '(m b) ... -> m b ...', m = num_batch_images)


        # contrastive loss

        """
        m - num batches of text (for multiview)
        n - num batches of images (for multiview)
        x - batches of text
        y - batches of images
        t - sequence dimension along text tokens
        i - sequence dimension along image tokens
        """

        text_to_image = einsum('m t d, n i d -> m n t i', text_latents_gather, image_latents_gather) * temp
        image_to_text = rearrange(text_to_image, '... t i -> ... i t')


        # calculate loss
        text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

        # print(f"shape of text to image: {text_to_image.shape}")


        # exponentiate
        text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

        # numerators
        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

        # denominator

        if self.decoupled_contrastive_learning:
            pos_mask = torch.eye(b, device = device, dtype = torch.bool)
            text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))

        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp))

        # loss

        text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim = -1)
        image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(dim = -1)

        # calculate CL loss

        cl_losses = (text_to_image_loss + image_to_text_loss) / 2 / bs_single_gpu


        cl_loss = cl_losses[0]

        loss_dict['cl_loss'] = cl_loss.item()


        loss_dict['cl_loss'] = cl_loss.item()
        return cl_loss, loss_dict



    def forward_old(
            self,
            text,
            image,
            device,
            return_loss = False,
            return_loss_dict = False,
            return_encodings = False,
            return_latents = False,
            use_seg=False, 
            seg_mask=None,
            seg_valid_mask=None,
            text_valid_mask=None,
            seg_weight=1.0,
            accelerator=None,
            freeze_image_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
            freeze_text_encoder = False,    # text encoder is not trained if this is set to True
            text_to_image = True,           # in the case the extra projection is turned on, would return different similarity values depending on modality directionality
            aug_text = None,                # augmented text (for multiview)
            aug_image = None                # augmented image (for multiview)
    ):
        """
        the original forward function, support mixed data types in a batch, but this is for old code support only!
        now more recommend to use forward_batch function
        """
        b, device = text.input_ids.shape[0], device

        # derive text mask

        text_mask =text.attention_mask

        self.seg_weight = seg_weight

        # ssl
        loss_dict = {}

        text_ssl_loss = 0
        image_ssl_loss = 0

        if return_loss:
            #print("-----------")
            #print(text.input_ids.shape)
            #print(text.attention_mask.shape)
            #print("------------")
            text_ssl_loss = self.mlm(text.input_ids, attention_mask = text.attention_mask) if self.use_mlm else 0
            # loss_dict['text_ssl_loss'] = text_ssl_loss.item() if self.use_mlm else 0
            image_ssl_loss = self.visual_ssl(image) if self.use_visual_ssl else 0
            # loss_dict['image_ssl_loss'] = image_ssl_loss.item() if self.use_visual_ssl else 0

        # concat augmented texts and images and do some asserts

        num_batch_texts = num_batch_images = 1

        if exists(aug_text):
            aug_text = cast_tuple(aug_text)
            assert all(map(lambda t: t.shape == text.shape, aug_text))
            num_batch_texts = len(aug_text) + 1

            aug_text = torch.cat(aug_text, dim = 0)

            aug_text_mask = aug_text != self.text_pad_id

            text_mask = torch.cat((text_mask, aug_text_mask), dim = 0)
            text = torch.cat((text, aug_text), dim = 0)

        if exists(aug_image):
            aug_image = cast_tuple(aug_image)
            assert all(map(lambda i: i.shape == image.shape, aug_image))
            num_batch_images = len(aug_image) + 1

            aug_image = torch.cat(aug_image, dim = 0)

            image = torch.cat((image, aug_image), dim = 0)

        is_multiview = (num_batch_texts > 1 or num_batch_images > 1)
        #assert not (return_loss and not self.training), 'loss cannot be used if not training'
        assert not (not return_loss and is_multiview), 'do not pass in augmented texts or images if not training'
        assert not (self.multiview_loss_weight == 0 and is_multiview), 'multiview loss weight cannot be 0 if augmented text or images passed in'

        # get encoded text

        text_args = (text.input_ids,text.attention_mask)

        if not self.text_encode_without_mask:
            text_args = (*text_args, text_mask)


        text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
        enc_text = text_embeddings[0]

        # depending on whether text is using causal mask, post process, moving eos token to the first position

        if self.text_causal_mask:
            eos_text_mask = (text == self.text_eos_id)
            assert torch.all(torch.any(eos_text_mask, dim = -1)), f'some of the text rows does not have the eos id {self.text_eos_id}'

            text_len = text.shape[-1]
            eos_indices = eos_text_mask.float().argmax(dim = -1, keepdim = True)

            eos_text_mask = torch.zeros_like(eos_text_mask).scatter(1, eos_indices, 1.).bool()
            eos_text_mask = rearrange(eos_text_mask, '... -> ... 1')

            eos_tokens = enc_text.masked_select(eos_text_mask)
            rest_tokens = enc_text.masked_select(~eos_text_mask)

            eos_tokens = rearrange(eos_tokens, '(b d) -> b 1 d', b = b)
            rest_tokens = rearrange(rest_tokens, '(b n d) -> b n d', b = b, n = text_len - 1)
            enc_text = torch.cat((eos_tokens, rest_tokens), dim = 1)

        # whether to train image encoder, in the case that the image net was pretrained as recommended in LiT

        """enc_image = model_forward_with_context(
            fn = self.visual_transformer,
            args = (image,),
            freeze = freeze_image_encoder
        )"""

        # print(f"image shape: {image.shape}")
        B, C, D, W, H = image.shape

        enc_image= self.visual_transformer(image, return_encoded_tokens=True)

        # print(f"encoded image shape: {enc_image.shape}")

        if use_seg:
            # use the seg valid mask to choose the image to be segmented
            print(f"seg_valid_mask: {seg_valid_mask}, enc image shape: {enc_image.shape}")
            enc_seg_image = enc_image[seg_valid_mask.squeeze(1).bool()]
            print(f"enc seg image shape: {enc_seg_image.shape}")
            seg_mask = seg_mask[seg_valid_mask.squeeze(1).bool()]
            if enc_seg_image.shape[0] == 0: # when seg 1 and report 3, causes memory issues, tmp fix todo: refractor the code
                seg_loss = 0.
                loss_dict['seg_loss'] = 0.
            else:
                # due to memory issues, use one for seg only now
                seg_mask = seg_mask.float().to(device)[0:1]
                enc_seg_image = enc_seg_image[0:1]
                b, d, w, h, c = enc_seg_image.shape
                p_h, p_w, p_d = H//h, W//w, D//d
                tokens_to_seg = enc_seg_image.reshape(-1, c) # b, l, c -> b*l, c
                # use the linear head for 
                seg_logits = self.visual_transformer.seg_head(tokens_to_seg)
                # reshape the logits to the original shape, with each pixel
                seg_preds = seg_logits.reshape(b, d, w, h, p_d, p_w, p_h, -1)
                seg_preds = seg_preds.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, -1, D, W, H)
                # seg_mask = seg_mask.float().to(device)
                seg_loss = self.seg_criterion(seg_preds, seg_mask)
                loss_dict['seg_loss'] = seg_loss.item()


        #print("This is visual encoding")
        global h_r, w_r, z_r
        h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]

        #enc_image, max_indices = torch.max(enc_image, dim=1)
        enc_image_send = enc_image

        enc_image = torch.mean(enc_image, dim=1)

        #kernel_size = (enc_image.size(1), enc_image.size(2), enc_image.size(3))

        #enc_image = enc_image.permute(0,4,1,2,3)
        # Perform max pooling over dimensions 1, 2, and 3
        #enc_image = F.max_pool3d(enc_image, kernel_size=kernel_size)

        #enc_image = enc_image.permute(0,2,3,4,1)

        #print(enc_image.shape, flush=True)
        #enc_image = enc_image[:,0,:]
        #print(enc_image.shape, flush=True)
        # print("test all pooling")
    

        enc_image = enc_image.view(enc_image.shape[0], -1)

       # print(enc_image.shape, flush=True)

        # early return of encodings, if needed (for DALL-E2)

        if return_encodings:
            return enc_text, enc_image

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only

        # if self.use_all_token_embeds:
        #     assert enc_text.ndim == 3, 'encoded text must have 3 dimensions (batch, seq, features)'
        #     assert enc_image.ndim == 3, 'encoded image must have 3 dimensions (batch, seq [height x width], features)'
        #     text_embeds = enc_text[:, 1:] if self.text_has_cls_token else enc_text
        #     image_embeds = enc_image[:, 1:] if self.visual_has_cls_token else enc_image
        # else:

        text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
        image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image

        # project to latents
        #text_embeds = text_embeds.view(text_embeds.shape[0], -1)
        text_embeds = text_embeds[:,0,:]

        # print(f"text embeds shape: {text_embeds.shape}")
        # print(f"image embeds shape: {image_embeds.shape}")
        # print(f"text valid mask: {text_valid_mask}")

        # select those with valid reports only for contrastive learning
        text_embeds = text_embeds[text_valid_mask.squeeze(1).bool(), :]
        image_embeds = image_embeds[text_valid_mask.squeeze(1).bool(), :]

        print(f"after, text embeds shape: {text_embeds.shape}")

        if text_embeds.shape[0] <= 1: # when 1, no contrastive learning
            loss = seg_loss * self.seg_weight

            loss_dict['loss_total'] = loss.item()

            if not return_loss_dict:
                return loss
            else:
                return loss, loss_dict
    
        else:
        #text_embeds = torch.mean(text_embeds, dim=1)
            text_latents = self.to_text_latent(text_embeds)

            image_latents = self.to_visual_latent(image_embeds)



            text_latents, image_latents = map(l2norm, (text_latents, image_latents))

            # print(f"shape of text latents: {text_latents.shape}, shape of image latents: {image_latents.shape}")
            # print(f"device of text latents: {text_latents.device}, device of image latents: {image_latents.device}")

            

            # exit()

            # calculate another set of latents for image to text (vs text to image)
            # proposed by CLOOB

            text_latents_extra, image_latents_extra = text_latents, image_latents
            # if self.extra_latent_projection:
            #     text_latents_extra = self.to_text_latent_extra(text_embeds)
            #     image_latents_extra = self.to_visual_latent_extra(image_embeds)
            #     text_latents_extra, image_latents_extra = map(l2norm, (text_latents_extra, image_latents_extra))

            # whether to early return latents

            if return_latents:
                # if self.extra_latent_projection:
                #     return text_latents, image_latents, text_latents_extra, image_latents_extra

                return text_latents, image_latents, enc_image_send

            # get temperature

            temp = self.temperature.exp()

            # early return, if needed


            # if not return_loss and self.use_all_token_embeds:
            #     einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
            #     return einsum('b d, b i d -> b t i', *einsum_args) * temp

            if not return_loss: # and not self.use_all_token_embeds:
                einsum_args = (text_latents, image_latents)
                return einsum('b d, b d -> b', *einsum_args) * temp

            # split out multiview dimension for text and images

            bs_single_gpu = text_latents.shape[0]

            # gather
            assert accelerator is not None, "accelerator is not provided"

            text_latents_gather = AllGather.apply(text_latents, accelerator)
            image_latents_gather = AllGather.apply(image_latents, accelerator)

            text_latents_gather = rearrange(text_latents_gather, '(m b) ... -> m b ...', m = num_batch_texts)
            image_latents_gather = rearrange(image_latents_gather, '(m b) ... -> m b ...', m = num_batch_images)
            
            

            # if self.extra_latent_projection:
            #     text_latents_extra = rearrange(text_latents_extra, '(m b) ... -> m b ...', m = num_batch_texts)
            #     image_latents_extra = rearrange(image_latents_extra, '(m b) ... -> m b ...', m = num_batch_images)

            # contrastive loss

            """
            m - num batches of text (for multiview)
            n - num batches of images (for multiview)
            x - batches of text
            y - batches of images
            t - sequence dimension along text tokens
            i - sequence dimension along image tokens
            """

            # if self.use_all_token_embeds:
            #     # fine-grained CLIP logic
            #     sim_text_to_image = einsum('m x t d, n y i d -> m n x y t i', text_latents, image_latents) * temp

            #     sim_image_to_text = sim_text_to_image
            #     if self.extra_latent_projection:
            #         sim_image_to_text = einsum('m x t d, n y i d -> m n x y t i', text_latents_extra, image_latents_extra) * temp

            #     text_to_image = reduce(sim_text_to_image, '... t i -> ... t', 'max')
            #     text_to_image_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t', m = num_batch_texts).bool()
            #     text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)

            #     image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m = num_batch_texts).bool()
            #     masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
            #     image_to_text = reduce(reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')
            # else:

            text_to_image = einsum('m t d, n i d -> m n t i', text_latents_gather, image_latents_gather) * temp
            image_to_text = rearrange(text_to_image, '... t i -> ... i t')

                # if self.extra_latent_projection:
                #     image_to_text = einsum('m t d, n i d -> m n i t', text_latents_extra, image_latents_extra) * temp

            # calculate loss

            text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
            image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

            print(f"shape of text to image: {text_to_image.shape}")


            # exponentiate
            text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

            # numerators
            text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

            # denominator

            if self.decoupled_contrastive_learning:
                pos_mask = torch.eye(b, device = device, dtype = torch.bool)
                text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))

            text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp))

            # loss

            text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim = -1)
            image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(dim = -1)

            # loss_dict['text_to_image_loss'] = text_to_image_loss.item()
            # loss_dict['image_to_text_loss'] = image_to_text_loss.item()

            # calculate CL loss

            cl_losses = (text_to_image_loss + image_to_text_loss) / 2 / bs_single_gpu

            # loss_dict['cl_loss_total'] = cl_losses.item()

            # get main CL loss vs multiview CL losses

            cl_loss, multiview_cl_loss = cl_losses[0], cl_losses[1:]

            loss_dict['cl_loss'] = cl_loss.item()
            # loss_dict['multiview_cl_loss'] = multiview_cl_loss.mean().item() if len(multiview_cl_loss) > 0 else 0

            # if no augmented text or images passed in, multiview loss weight is 0

            multiview_loss_weight = self.multiview_loss_weight if is_multiview else 0

            # calculate weights

            cl_loss_weight = 1 - (self.text_ssl_loss_weight + self.image_ssl_loss_weight + multiview_loss_weight)
        
            loss = (cl_loss * cl_loss_weight) \
                + (text_ssl_loss * self.text_ssl_loss_weight) \
                + (image_ssl_loss * self.image_ssl_loss_weight) \
                    + (seg_loss * self.seg_weight)

            loss_dict['loss_total'] = loss.item()

            # add multiview CL loss with weight

            if is_multiview:
                loss = loss + multiview_cl_loss.mean() * multiview_loss_weight

            if not return_loss_dict:
                return loss
            else:
                return loss, loss_dict







