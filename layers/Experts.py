import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


class FourierFilterExpert(nn.Module):
    def __init__(self, d_model, seq_len_patches):
        super(FourierFilterExpert, self).__init__()
        self.d_model = d_model

        self.freq_dim = seq_len_patches // 2 + 1

        # Store real and imaginary parts in the last dimension.
        self.complex_weight = nn.Parameter(
            torch.randn(self.freq_dim, d_model, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        """Input and output: (B*C, N, D)."""
        B, N, D = x.shape

        x_fft = torch.fft.rfft(x, dim=1)  # (B, freq_dim, D)

        # Match the available input and filter frequency bins.
        curr_freq_dim = x_fft.shape[1]
        weight = torch.view_as_complex(self.complex_weight)  # (freq_dim, D)

        if curr_freq_dim > self.freq_dim:
            x_fft = x_fft[:, : self.freq_dim, :]
        elif curr_freq_dim < self.freq_dim:
            weight = weight[:curr_freq_dim, :]

        y_fft = x_fft * weight

        # Restore the original patch count, including odd lengths.
        y = torch.fft.irfft(y_fft, n=N, dim=1)

        return y


class DilatedTCNExpert(nn.Module):
    def __init__(self, d_model, dilation_factors=[1, 2, 4]):
        super(DilatedTCNExpert, self).__init__()
        self.layers = nn.ModuleList()
        self.d_model = d_model

        for d in dilation_factors:
            self.layers.append(self._build_block(d_model, d))

    def _build_block(self, d_model, dilation):
        # Left padding preserves causality and patch count.
        padding_size = (3 - 1) * dilation

        return nn.ModuleDict(
            {
                'pad': nn.ConstantPad1d((padding_size, 0), 0),
                'conv_filter': nn.Conv1d(
                    d_model, d_model, kernel_size=3, dilation=dilation
                ),
                'conv_gate': nn.Conv1d(
                    d_model, d_model, kernel_size=3, dilation=dilation
                ),
                'drop': nn.Dropout(0.1),
                'proj': nn.Conv1d(d_model, d_model, kernel_size=1),
            }
        )

    def forward(self, x):
        # (B, N, D) -> (B, D, N)
        x = x.transpose(1, 2)

        for layer in self.layers:
            residual = x

            out = layer['pad'](x)

            filter_out = layer['conv_filter'](out)
            gate_out = layer['conv_gate'](out)

            out = F.gelu(filter_out) * torch.sigmoid(gate_out)

            out = layer['drop'](out)
            out = layer['proj'](out)

            x = out + residual

        return x.transpose(1, 2)


class FeatureMLPExpert(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super(FeatureMLPExpert, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)
