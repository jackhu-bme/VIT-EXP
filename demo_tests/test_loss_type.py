# check the loss implementations between the one in eva (clip loss) and the one in ct clip (seem not to be clip loss but something close to that)
# start with image features and text features

from clip_loss import ClipLoss

import torch

from einops import rearrange

from torch import nn, einsum

# the one in eva

x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
y = torch.tensor([[0.2, 0.3], [0.3, 0.6], [0.4, 0.9], [0.2, 0.5]])

loss_fn_1 = ClipLoss(local_loss=False, gather_with_grad=True, cache_labels=False, rank=0, world_size=1, use_horovod=False, smoothing=0.)

loss_1 = loss_fn_1(x, y)

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# here is the loss from the ct clip work

def ct_clip_loss(text_latents_gather, image_latents_gather):
    temp = 1 # attention: original implementation uses a learnable parameter, which is strange!!!
    text_latents_gather = rearrange(text_latents_gather, '(m b) ... -> m b ...', m = 1)
    image_latents_gather = rearrange(image_latents_gather, '(m b) ... -> m b ...', m = 1)
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
    print(f"in ct-clip, text to image:{text_to_image}")
    image_to_text = rearrange(text_to_image, '... t i -> ... i t')
    # calculate loss
    text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
    image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')
    # print(f"shape of text to image: {text_to_image.shape}")
    # exponentiate
    text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))
    print(f"after exp, in ct-clip text to image: {text_to_image_exp}")
    # numerators
    text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))
    print(f"after diag, in ct-clip text to image: {text_to_image_pos}")
    # denominator
    text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp))

    # loss

    text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim = -1)
    image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(dim = -1)

    # calculate CL loss

    bs_single_gpu = text_latents_gather.shape[1]

    cl_losses = (text_to_image_loss + image_to_text_loss) / 2 # / bs_single_gpu

    return cl_losses

loss_fn_2 = ct_clip_loss

loss_2 = loss_fn_2(x, y)


print(f"loss_1: {loss_1}, loss_2: {loss_2}")








