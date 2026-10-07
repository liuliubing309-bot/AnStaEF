import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PatchEmbedding, CrossVariableContextExchange
from layers.RevIN import RevIN
from layers.Router import StatisticsGuidedTopKRouter
from layers.Anchor import AnchorStream
from layers.Experts import FourierFilterExpert, DilatedTCNExpert, FeatureMLPExpert


class SegmentLevelTemporalInteraction(nn.Module):
    def __init__(self, d_model, segment_size=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.segment_size = max(1, segment_size)
        self.num_heads = num_heads if d_model % num_heads == 0 else 1
        self.head_dim = d_model // self.num_heads
        self.scale = self.head_dim**-0.5

        self.seg_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        bsz, n_tokens, d_model = x.shape
        seg_size = self.segment_size
        n_segments = (n_tokens + seg_size - 1) // seg_size
        padded_tokens = n_segments * seg_size
        pad_len = padded_tokens - n_tokens

        if pad_len > 0:
            x_pad = torch.cat([x, x[:, -1:, :].expand(-1, pad_len, -1)], dim=1)
        else:
            x_pad = x

        seg_tokens = x_pad.reshape(bsz, n_segments, seg_size, d_model).mean(dim=2)
        seg_tokens = self.seg_norm(seg_tokens)

        q = (
            self.q_proj(seg_tokens)
            .reshape(bsz, n_segments, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(seg_tokens)
            .reshape(bsz, n_segments, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(seg_tokens)
            .reshape(bsz, n_segments, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        seg_context = (
            torch.matmul(attn, v)
            .transpose(1, 2)
            .contiguous()
            .reshape(bsz, n_segments, d_model)
        )
        seg_context = self.drop(self.out_proj(seg_context))

        patch_context = (
            seg_context.unsqueeze(2)
            .expand(bsz, n_segments, seg_size, d_model)
            .reshape(bsz, padded_tokens, d_model)
        )
        fusion_gate = torch.sigmoid(
            self.gate_proj(torch.cat([x_pad, patch_context], dim=-1))
        )
        y = self.out_norm(x_pad + fusion_gate * patch_context)

        if pad_len > 0:
            y = y[:, :n_tokens, :]
        return y


class ForecastHead(nn.Module):
    def __init__(
        self, temporal_mixer_type, temporal_mixer, pred_len, head, temporal_norm=None
    ):
        super().__init__()
        self.temporal_mixer_type = temporal_mixer_type
        self.temporal_mixer = temporal_mixer
        self.temporal_norm = temporal_norm
        self.pred_len = pred_len
        self.head = head
        self.head_in_features = head.in_features

    def forward(self, z):
        if self.temporal_mixer_type == 'segment_attn':
            z = self.temporal_mixer(z)
        else:
            z_mixed = self.temporal_mixer(z.transpose(1, 2)).transpose(1, 2)
            z = self.temporal_norm(z + z_mixed)
        z_flat = z.reshape(z.shape[0], -1)

        if z_flat.shape[1] != self.head_in_features:
            raise RuntimeError(
                f"ForecastHead expected flattened dim {self.head_in_features}, got {z_flat.shape[1]}. "
                f"Check num_tokens={z.shape[1]}, d_model={z.shape[2]}, patch_len/stride settings, and temporal mixer output shape."
            )

        return self.head(z_flat)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.d_model = args.d_model
        self.temporal_mixer_type = getattr(args, 'temporal_mixer_type', 'segment_attn')

        self.num_patches = int((self.seq_len - self.patch_len) / self.stride) + 1

        self.revin = RevIN(1, affine=True)

        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=args.dropout,
        )

        self.pre_norm = nn.LayerNorm(self.d_model)

        self.use_cvce = bool(getattr(args, 'use_cvce', 1))
        if self.use_cvce:
            n_vars = getattr(args, 'enc_in', 7)
            self.cvce = CrossVariableContextExchange(
                d_model=self.d_model,
                n_vars=n_vars,
                bottleneck=getattr(args, 'cvce_bottleneck', 4),
                num_heads=getattr(args, 'cvce_heads', 4),
                dropout=args.dropout,
            )

        self.router = StatisticsGuidedTopKRouter(
            d_model=self.d_model,
            num_experts=args.num_experts,
            patch_len=self.patch_len,
            top_k=args.top_k,
            temperature=getattr(args, 'router_temperature', 1.0),
            router_jitter=getattr(args, 'router_jitter', 0.0),
            smoothing=getattr(args, 'router_smoothing', 0.02),
        )

        if self.args.use_anchor:
            self.anchor = AnchorStream(d_model=self.d_model)

        self.shared_experts = nn.ModuleList(
            [
                FourierFilterExpert(self.d_model, self.num_patches),
                DilatedTCNExpert(self.d_model),
                FeatureMLPExpert(self.d_model),
            ]
        )
        self.expert_type_ids = [i % 3 for i in range(args.num_experts)]
        self.expert_scale = nn.Parameter(torch.ones(args.num_experts, self.d_model))
        self.expert_bias = nn.Parameter(torch.zeros(args.num_experts, self.d_model))
        if self.temporal_mixer_type == 'segment_attn':
            temporal_mixer = SegmentLevelTemporalInteraction(
                d_model=self.d_model,
                segment_size=getattr(args, 'segment_size', 4),
                num_heads=getattr(args, 'segment_attn_heads', 4),
                dropout=args.dropout,
            )
            temporal_norm = None
        else:
            temporal_mixer = nn.Sequential(
                nn.Conv1d(
                    self.d_model,
                    self.d_model,
                    kernel_size=3,
                    padding=1,
                    groups=self.d_model,
                ),
                nn.GELU(),
                nn.Conv1d(self.d_model, self.d_model, kernel_size=1),
                nn.Dropout(args.dropout),
            )
            temporal_norm = nn.LayerNorm(self.d_model)

        self.head_nf = self.d_model * self.num_patches
        self.forecast_head = ForecastHead(
            temporal_mixer_type=self.temporal_mixer_type,
            temporal_mixer=temporal_mixer,
            pred_len=self.pred_len,
            head=nn.Linear(self.head_nf, self.pred_len),
            temporal_norm=temporal_norm,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: (Batch, Seq_Len, n_vars)
        """
        B, L, N_vars = x_enc.shape

        x = x_enc.permute(0, 2, 1).contiguous().reshape(B * N_vars, L, 1)

        x = self.revin(x, 'norm')

        x_embed, raw_patches = self.patch_embedding(x, return_raw_patches=True)
        x_embed = self.pre_norm(x_embed)

        if self.use_cvce and N_vars > 1:
            x_embed = self.cvce(x_embed, B, N_vars)

        top_k_probs, top_k_indices, router_logits = self.router(x_embed, raw_patches)

        if self.args.use_anchor:
            anchor_out = self.anchor(x_embed)
        else:
            anchor_out = 0

        B_tot, N, K = top_k_indices.shape
        D = self.d_model

        if self.training:
            candidate_outputs = []
            for i in range(self.args.num_experts):
                base_out = self.shared_experts[self.expert_type_ids[i]](
                    x_embed
                )  # (B*C, N, D)
                scale = self.expert_scale[i].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                bias = self.expert_bias[i].unsqueeze(0).unsqueeze(0)
                out = base_out * scale + bias
                candidate_outputs.append(out.unsqueeze(2))

            candidate_pool = torch.cat(candidate_outputs, dim=2)  # (B*C, N, M, D)
            gather_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, D)
            selected_candidate_outputs = torch.gather(
                candidate_pool, dim=2, index=gather_indices
            )
        else:
            candidate_pool = None

            expert_type_map = top_k_indices.new_tensor(self.expert_type_ids)
            selected_type_ids = expert_type_map[top_k_indices]

            # Selected candidate parameters: (B*C, N, K, D)
            selected_scales = self.expert_scale[top_k_indices]
            selected_biases = self.expert_bias[top_k_indices]
            selected_candidate_outputs = torch.zeros_like(selected_scales)

            for type_id, base_expert in enumerate(self.shared_experts):
                type_mask = selected_type_ids.eq(type_id).unsqueeze(
                    -1
                )  # (B*C, N, K, 1)
                if not torch.any(type_mask):
                    continue

                base_out = base_expert(x_embed).unsqueeze(2)  # (B*C, N, 1, D)
                modulated = base_out * selected_scales + selected_biases
                selected_candidate_outputs = torch.where(
                    type_mask, modulated, selected_candidate_outputs
                )

        fusion_weights = top_k_probs.unsqueeze(-1)
        expert_stream_output = torch.sum(
            selected_candidate_outputs * fusion_weights, dim=2
        )

        if self.args.use_anchor:
            z = expert_stream_output + anchor_out
        else:
            z = expert_stream_output
        y_pred = self.forecast_head(z)

        y_pred = y_pred.unsqueeze(-1)
        y_pred = self.revin(y_pred, 'denorm')
        y_pred = y_pred.reshape(B, N_vars, self.pred_len).permute(0, 2, 1)

        if self.training:
            return y_pred, router_logits, candidate_pool
        else:
            return y_pred, None, None
