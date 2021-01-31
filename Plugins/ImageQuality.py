import torch

def PSNR(x_pre,x_true):
    psnr_ = 10*torch.log10((x_true.max()**2)/((x_pre-x_true)**2).mean())
    return psnr_
