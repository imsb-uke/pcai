from typing import Optional

import torch
import torch.nn as nn


class BagSelfAttentionModule(nn.Module):
    """Self-attention module.

    Implemented as proposed in https://github.com/gmum/Kernel_SA-AbMILP.

    Expects latent input of shape BxNxL and outputs latents of same shape BxNxL.
    """

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: Optional[int] = None,
        gamma_init_value: int = 1,
        gamma_trainable: bool = False,
    ):

        super().__init__()

        if hidden_size is None:
            hidden_size = input_size // 8

        self.query_conv = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=1
        )
        self.key_conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        if gamma_init_value == 0:
            gamma = torch.zeros(1)
        else:
            gamma = torch.ones(1) * gamma_init_value

        if gamma_trainable:
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("gamma", gamma)

    def forward(self, inputs: torch.Tensor):
        # in: BxNxL

        x = inputs.permute((0, 2, 1))
        # channels 2 and 1 need to be flipped for 1x1 conv to be the same as linear layer with same in_dim !

        # https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-vs-linear-layer
        # nn.Conv1d with a kernel size of 1 and nn.Linear give essentially the same results.
        # these are basically linear layers that process each view separately
        proj_query = self.query_conv(x).permute(0, 2, 1)
        proj_key = self.key_conv(x)

        energy = torch.bmm(proj_query, proj_key)

        self.attention = self.softmax(energy)

        proj_value = self.value_conv(x)

        out = torch.bmm(proj_value, self.attention.permute(0, 2, 1))

        out = self.gamma * out + x

        out = out.permute((0, 2, 1))

        # out: BxNxL
        return out
