import torch
import torch.nn as nn
import torch.nn.functional as F


class StatisticalFeatureExtractor(nn.Module):
    def __init__(self, patch_len, eps=1e-5):
        super().__init__()
        self.patch_len = patch_len
        self.eps = eps

        indices = torch.arange(patch_len).float()
        self.register_buffer('time_indices', indices.view(1, 1, -1))

        self.x_mean = indices.mean()
        self.x_var = torch.var(indices, unbiased=False)

    def forward(self, x):
        # Trend slope, bounded by tanh.
        mean_y = x.mean(dim=-1, keepdim=True)
        x_centered = self.time_indices - self.x_mean
        y_centered = x - mean_y

        covariance = (x_centered * y_centered).mean(dim=-1)
        slope = covariance / (self.x_var + self.eps)
        trend_strength = torch.tanh(slope).unsqueeze(-1)

        # Log-standard-deviation descriptor.
        std_y = x.std(dim=-1, unbiased=False)
        volatility = torch.log1p(std_y).unsqueeze(-1)
        volatility = torch.clamp(volatility, max=10.0)

        # Lag-1 autocorrelation with patch-wise centering.
        y_centered_current = y_centered[:, :, 1:]
        y_centered_lag = y_centered[:, :, :-1]

        numerator = (y_centered_current * y_centered_lag).mean(dim=-1)
        denominator = y_centered.var(dim=-1, unbiased=False) + self.eps
        autocorr = (numerator / denominator).unsqueeze(-1)

        # (B*C, N, 3): trend, volatility, autocorrelation
        return torch.cat([trend_strength, volatility, autocorr], dim=-1)


class StatisticsGuidedTopKRouter(nn.Module):
    def __init__(
        self,
        d_model,
        num_experts,
        patch_len=None,
        top_k=2,
        d_stat_hidden=16,
        temperature=1.0,
        router_jitter=0.0,
        smoothing=0.02,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.temperature = max(1e-3, float(temperature))
        self.router_jitter = max(0.0, float(router_jitter))
        self.smoothing = min(1.0, max(0.0, float(smoothing)))
        self.feature_extractor = (
            StatisticalFeatureExtractor(patch_len=patch_len)
            if patch_len is not None
            else None
        )
        self.stat_projector = nn.Sequential(
            nn.Linear(3, d_stat_hidden),
            nn.ReLU(),
            nn.Linear(d_stat_hidden, d_stat_hidden),
        )

        input_dim = d_model + d_stat_hidden

        # Local Context Router over adjacent patches.
        self.context_gate = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Conv1d(input_dim, num_experts, kernel_size=1),
        )

        self.noise_generator = nn.Conv1d(input_dim, num_experts, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, x_embed, x_raw_stats):
        expected_patch_len = (
            self.feature_extractor.patch_len
            if self.feature_extractor is not None
            else None
        )
        if (
            self.feature_extractor is not None
            and x_raw_stats.shape[-1] == expected_patch_len
        ):
            x_raw_stats = self.feature_extractor(x_raw_stats)
        elif x_raw_stats.shape[-1] != 3:
            if expected_patch_len is None:
                raise ValueError(
                    f"Expected precomputed stats with last dim 3, got {tuple(x_raw_stats.shape)}"
                )
            raise ValueError(
                f"Expected raw patches with length {expected_patch_len} or precomputed stats with last dim 3, got {tuple(x_raw_stats.shape)}"
            )

        stats_emb = self.stat_projector(x_raw_stats)

        router_input = torch.cat([x_embed, stats_emb], dim=-1)
        if self.training and self.router_jitter > 0:
            router_input = (
                router_input + torch.randn_like(router_input) * self.router_jitter
            )

        router_input_t = router_input.transpose(1, 2)

        clean_logits = self.context_gate(router_input_t).transpose(1, 2)  # (B*C, N, M)

        noise_logits = self.noise_generator(router_input_t).transpose(1, 2)

        if self.training:
            raw_noise_stddev = self.softplus(noise_logits)
            standard_normal_noise = torch.randn_like(clean_logits)
            logits = clean_logits + standard_normal_noise * raw_noise_stddev
        else:
            logits = clean_logits
        logits = logits / self.temperature

        # Normalize over selected candidates only.
        k = min(self.top_k, self.num_experts)
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        if self.training and self.smoothing > 0:
            top_k_probs = (1.0 - self.smoothing) * top_k_probs + (
                self.smoothing / float(k)
            )
        return top_k_probs, top_k_indices, logits
