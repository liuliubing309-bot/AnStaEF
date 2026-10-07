import torch
import torch.nn as nn


class AnchorStream(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv_a = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding
        )
        self.conv_b = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding
        )
        self.conv_c = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding
        )
        self.activation = nn.Identity()

    def forward(self, x):
        """Input and output: (B*C, N, D)."""
        x_in = x.transpose(1, 2)

        common_term = self.conv_a(x_in)

        interaction_term = self.conv_b(x_in) * self.conv_c(x_in)

        out = common_term + interaction_term

        return out.transpose(1, 2)
