from segmentation_models_pytorch.losses import TverskyLoss

import torch

import torch.nn.functional as F

alpha = 0.5
beta = 0.5
gamma = 1.0
smooth = 0

tversky_criterion = TverskyLoss(mode="binary", alpha=alpha, beta=beta, smooth=smooth, gamma=gamma, from_logits=False)

def dice_loss(pred, target, smooth = 1e-4):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth)))
    
    return loss.mean()


# B=1, C=2, L=4
p = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]])
targets = torch.tensor([[[1, 0, 1, 0], [0, 1, 0, 1]]])

print(f"p shape: {p.shape}")
print(f"targets shape: {targets.shape}")

loss_t = tversky_criterion.forward(p, targets)

print(f"loss tversky: {loss_t}")

probs = p  # F.sigmoid(p)

loss_d = dice_loss(probs, targets)

print(f"loss dice: {loss_d}")


