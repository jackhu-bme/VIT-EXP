from segmentation_models_pytorch.losses import TverskyLoss

alpha = 0.5
beta = 0.5
gamma = 1.0
smooth = 1e-4

tversky_criterion = TverskyLoss(mode="binary", alpha=alpha, beta=beta, smooth=smooth, gamma=gamma)







