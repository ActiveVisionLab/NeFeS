import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x):
        x=x.clamp(-9.7, 11.08) # safer, ref: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        # return g * torch.exp(x.clamp(-15, 15))
        # print("g min: ", g.min(), "g max:", g.max())
        return g * torch.exp(x.clamp(-9.7, 11.08)) # clamp to avoid overflow

trunc_exp = _trunc_exp.apply

def trunc_softplus(x):
    x= x.clamp(-9.7, 11.08) # safer
    return torch.nn.functional.softplus(x)