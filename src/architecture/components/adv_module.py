import torch
import torch.nn as nn


class AdversarialModule(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
    ):

        super().__init__()
        
        self.alpha = alpha
        self.grad_reverse = GradReverse()

    def forward(self, inputs: torch.Tensor):
        return self.grad_reverse.apply(inputs, self.alpha)  # out: BxL


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
