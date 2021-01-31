import torch
from torch.autograd import Function
from Plugins import Config
# from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
# from torch.utils.dlpack import from_dlpack
from . import CTOperator

class SARTFunction(Function):
    def __init__(self):
        super(SARTFunction, self).__init__()

    @staticmethod
    def forward(ctx,input,proj):
        batchsize = input.shape[0]
        ctx.save_for_backward(input)
        if (input.shape[1] != 1):
            raise NotImplementedError
        sp = Config.getScanParam()
        out = input.clone()
        for i in range(batchsize):
            cupy_input = fromDlpack(to_dlpack(out[i,0,:,:]))
            cupy_proj = fromDlpack(to_dlpack(proj[i,0,:,:]))
            CTOperator.SART2D(cupy_proj,sp,sp['order'],cupy_input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        batchsize = grad_output.shape[0]
        # input = ctx.saved_varibles
        if (grad_output.shape[1] != 1):
            raise NotImplementedError
        sp = Config.getScanParam()
        grad = grad_output.clone()
        for i in range(batchsize):
            cupy_grad = fromDlpack(to_dlpack(grad[i, 0, :, :]))
            CTOperator.SART2DBackWard(cupy_grad,sp['order'],sp)
        return grad,None

class SARTlayer(torch.nn.Module):
    def __init__(self):
        super(SARTlayer, self).__init__()

    def forward(self,input,proj):
        return SARTFunction.apply(input,proj)
