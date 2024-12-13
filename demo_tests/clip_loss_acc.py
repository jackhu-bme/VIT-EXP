from timm.loss import LabelSmoothingCrossEntropy


import torch

import torch.nn as nn

from torch.nn import functional as F


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, accelerator):
        x_gather = accelerator.gather(x)
        ctx.num_processes = accelerator.num_processes
        ctx.process_index = accelerator.process_index
        return x_gather

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.chunk(ctx.num_processes, dim = 0)[ctx.process_index]
        return grad_input, None

class ClipLossAcc(nn.Module):
    def __init__(
            self,
            smoothing=0.,
    ):
        super().__init__()
        self.label_smoothing_cross_entropy = LabelSmoothingCrossEntropy(smoothing=smoothing) if smoothing > 0 else None

    def forward(self, image_features, text_features, accelerator, logit_scale=1.):
        device = image_features.device
        all_image_features = AllGather.apply(
            image_features, accelerator)
        all_text_features = AllGather.apply(text_features, accelerator)
        logits_per_image = logit_scale * all_image_features @ all_text_features.T
        logits_per_text = logits_per_image.T
        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        
        if self.label_smoothing_cross_entropy:
            total_loss = (
                self.label_smoothing_cross_entropy(logits_per_image, labels) +
                self.label_smoothing_cross_entropy(logits_per_text, labels)
                ) / 2
        else:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                ) / 2
            
        acc = None
        i2t_acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == labels).sum() / len(logits_per_text)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc}
        return total_loss #, acc
