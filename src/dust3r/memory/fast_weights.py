"""Pose fast weights: explicit associative updates, independently organized from the backbone.

The SwiGLU alignment update follows LaCT's analytic-gradient formulation.
Parameter names and packed weight ordering match existing Mem3R checkpoints.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .numerics import RMSNorm, inv_softplus, silu_backprop

class TTTMemory(nn.Module):
    def __init__(
            self,
            size, # Kept for API compatibility
            k_dim,
            v_dim,
            num_heads,
            inter_multi=1,
            base_lr=0.001,
            decay_scale=0.01,
            use_swiglu=True,  # Toggle: True=SwiGLU (3 weights), False=Linear MLP (1 weight)
            loss_fn="inner",
            **kwargs
        ) -> None:
        super().__init__()
        self.use_swiglu = use_swiglu
        self.decay_scale = decay_scale
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.loss_fn = loss_fn

        # Dimensions per head
        self.head_dim = v_dim // num_heads
        self.d_in = self.head_dim
        self.d_out = self.head_dim
        self.d_h = int(self.head_dim * inter_multi)

        # External projections
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.c_proj = nn.Linear(v_dim, v_dim, bias=False)
        self.o_norm = RMSNorm(self.head_dim, eps=1e-5)

        # Scale output projection near zero for stability
        nn.init.normal_(self.c_proj.weight, std=0.002)

        # Head-wise Learnable Decay Factor (Alpha)
        self.log_alpha_fc = nn.Linear(v_dim, num_heads, bias=False)
        nn.init.xavier_uniform_(self.log_alpha_fc.weight)

        # Dynamic Learning Rate Predictor
        # SwiGLU needs 3 LRs (w0, w1, w2) | Simple MLP needs 1 LR (w0)
        self.lr_count = 3 if use_swiglu else 1
        self.lr_fc = nn.Linear(v_dim, self.num_heads * self.lr_count)
        self.base_lr_inv = inv_softplus(base_lr)

        # Initialize Memory Weights
        gain = math.sqrt(2)
        # In SwiGLU: w0 maps to hidden dim (d_h). In Simple MLP: w0 maps to output dim (d_out).
        w0_out_dim = self.d_h if use_swiglu else self.d_out
        w0 = torch.randn(self.num_heads, self.d_in, w0_out_dim) * gain / math.sqrt(self.d_in)

        self.w0_size = w0.numel()

        if self.use_swiglu:
            w1 = torch.randn(self.num_heads, self.d_h, self.d_out) * gain / math.sqrt(self.d_h)
            w2 = torch.randn(self.num_heads, self.d_in, self.d_h) * gain / math.sqrt(self.d_in)
            self.w1_size = w1.numel()
            self.w2_size = w2.numel()
            # Pack [w0, w1, w2]
            packed = torch.cat([w0.flatten(), w1.flatten(), w2.flatten()]).view(1, 1, -1)
        else:
            self.w1_size = 0
            self.w2_size = 0
            # Pack [w0] only
            packed = w0.flatten().view(1, 1, -1)

        self.mem = nn.Parameter(packed, requires_grad=True)

    def unpack_mem(self, mem):
        """ Unpacks the flat memory parameter into weight matrices for the computation graph. """
        B = mem.shape[0]
        mem = mem.view(B, -1)
        w0_flat = mem[:, :self.w0_size]

        # Reshape w0 based on mode
        w0_out_dim = self.d_h if self.use_swiglu else self.d_out
        w0 = w0_flat.view(B, self.num_heads, self.d_in, w0_out_dim)

        if self.use_swiglu:
            w1_flat = mem[:, self.w0_size : self.w0_size + self.w1_size]
            w2_flat = mem[:, self.w0_size + self.w1_size:]
            w1 = w1_flat.view(B, self.num_heads, self.d_h, self.d_out)
            w2 = w2_flat.view(B, self.num_heads, self.d_in, self.d_h)
            return w0, w1, w2
        return w0, None, None

    def pack_mem(self, w0, w1=None, w2=None):
        """ Packs matrices back into a single flat tensor. """
        B = w0.shape[0]
        if self.use_swiglu:
            return torch.cat([w0.reshape(B, -1), w1.reshape(B, -1), w2.reshape(B, -1)], dim=1).view(B, 1, -1)
        return w0.reshape(B, -1).view(B, 1, -1)


    def _prepare_query(self, image_summary):
        projected = self.proj_q(image_summary)
        per_head = rearrange(projected, "b l (h d) -> (b h) l d",
                             h=self.num_heads, l=1)
        return per_head / (per_head.norm(dim=2, keepdim=True) + 1e-5)

    def _update_controls(self, image_summary):
        projected = self.proj_q(image_summary)
        retention = 1.0 - self.decay_scale * torch.sigmoid(self.log_alpha_fc(projected))
        retention = rearrange(retention, "b 1 h -> (b h) 1 1")
        with torch.autocast(device_type="cuda", enabled=False):
            rate_logits = self.lr_fc(projected.float())
        rates = F.softplus(rate_logits.float() + self.base_lr_inv)
        rates = rearrange(rates, "b l (cnt h) -> cnt (b h) l 1",
                          cnt=self.lr_count, h=self.num_heads)
        return retention, rates

    @staticmethod
    def _swiglu_directions(keys, targets, input_matrix, output_matrix,
                           value_matrix, squared_error):
        activation_input = keys @ input_matrix
        value_features = keys @ value_matrix
        activated = F.silu(activation_input)
        hidden_features = activated * value_features
        output_signal = targets - hidden_features @ output_matrix if squared_error else targets
        hidden_signal = output_signal @ output_matrix.transpose(-1, -2)
        output_direction = hidden_features.transpose(-1, -2) @ output_signal
        value_direction = keys.transpose(-1, -2) @ (hidden_signal * activated)
        input_direction = keys.transpose(-1, -2) @ silu_backprop(
            hidden_signal * value_features, activation_input)
        return input_direction, output_direction, value_direction

    @staticmethod
    def _advance_matrix(matrix, direction, rate, retention, max_update_norm):
        delta = direction * rate
        if max_update_norm > 0.0:
            magnitude = delta.norm(dim=(1, 2), keepdim=True) + 1e-7
            delta = delta * torch.minimum(torch.ones_like(magnitude), max_update_norm / magnitude)
        return matrix * retention + delta

    def update_mem(self, mem, feat_k, feat_v, grad_clip_norm=1.0):
        """Write a posterior target using alignment ascent (or negative L2 gradient).

        Targets are held fixed for the analytic inner partial derivative. The
        resulting update remains differentiable for outer-loop training.
        """
        batch_size = mem.shape[0]
        keys = self._prepare_query(feat_k)
        targets = rearrange(feat_v, "b l (h d) -> (b h) l d",
                            h=self.num_heads, l=1)
        retention, rates = self._update_controls(feat_k)
        packed_matrices = self.unpack_mem(mem)
        matrices = [rearrange(matrix, "b h d o -> (b h) d o")
                    for matrix in packed_matrices if matrix is not None]
        if self.use_swiglu:
            directions = self._swiglu_directions(keys, targets, *matrices,
                                                  squared_error=self.loss_fn == "l2")
        else:
            residual = targets - keys @ matrices[0]
            directions = (keys.transpose(-1, -2) @ residual,)
        advanced = [self._advance_matrix(matrix, direction, rates[index], retention,
                                         grad_clip_norm)
                    for index, (matrix, direction) in enumerate(zip(matrices, directions))]
        restored = [rearrange(matrix, "(b h) d o -> b h d o", b=batch_size)
                    for matrix in advanced]
        return self.pack_mem(*restored)

    def inquire(self, query, mem):
        """Read a pose prior, including head normalization and output projection."""
        batch_size = query.shape[0]
        keys = self._prepare_query(query)
        matrices = [rearrange(matrix, "b h d o -> (b h) d o")
                    for matrix in self.unpack_mem(mem) if matrix is not None]
        if self.use_swiglu:
            input_matrix, output_matrix, value_matrix = matrices
            values = (F.silu(keys @ input_matrix) * (keys @ value_matrix)) @ output_matrix
            values = self.o_norm(values)
        else:
            values = keys @ matrices[0]
        merged = rearrange(values, "(b h) l d -> b l (h d)",
                           h=self.num_heads, b=batch_size)
        return self.c_proj(merged)
