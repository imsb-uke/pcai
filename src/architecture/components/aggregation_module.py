import torch
import torch.nn as nn

class BagAggregationModule(nn.Module):
    """MIL aggregation module.

    Expects latent-patch input of shape BxNxL and outputs aggregated output of shape BxL.
    """

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 128,
    ):

        super().__init__()

        self.mil_layer = MILAttention(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs: torch.Tensor):
        # in: BxNxL

        """inputs shape: BxNxL this method only works for equal number of patches per batch."""

        # perform mil aggregation
        x, self.attention = self.mil_layer(inputs)
        # squeeze K dimension of 1 : BxKxL -> BxL
        x = torch.squeeze(x, 1)

        # out: BxL
        return x

class MILAttention(nn.Module):
    """
    multiple instance learning based on https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
    set k = 1 if only label for the entire bag is relevant, set k=n_patches/instances to get one prediction per instance

    input shape: BxNxL
    output shape: BxKxL
    """

    def __init__(self, input_size: int, hidden_size: int = 128, k: int = 1, **kwargs):
        super().__init__(**kwargs)

        self.attention_dense1 = nn.Linear(input_size, hidden_size)
        self.attention_tanh = nn.Tanh()
        self.attention_dense2 = nn.Linear(hidden_size, k)
        self.attention_softmax = nn.Softmax(dim=2)

    def forward(self, inputs: torch.Tensor):
        # FC layer is applied on last dimension of input only (can be otherwise arbitrary shape)
        alpha = self.attention_dense1(inputs)  # -> BxNxD
        alpha = self.attention_tanh(alpha)  # -> BxNxD
        alpha = self.attention_dense2(alpha)  # -> BxNxK
        alpha = torch.transpose(alpha, 2, 1)  # -> BxKxN

        alpha_sm = self.attention_softmax(alpha)  # -> BxKxN

        y = torch.bmm(alpha_sm, inputs)  # BxKxL

        return y, alpha
