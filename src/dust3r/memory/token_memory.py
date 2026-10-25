"""Transformer token memory retained for CUT3R baseline compatibility."""
import torch
from torch import nn
from dust3r.blocks import DecoderBlock

class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        ) # [1, 1, 768] pose mask token
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        ) # [1, 256, 1536] pose mem
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v, return_attn=False):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C] global_img_feat
        feat_v: [B, 1, C] out_pose_feat
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)

        attention_maps = []
        for blk in self.write_blocks:
            mem, _, self_attn, cross_attn = blk(mem, feat, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return mem

    def inquire(self, query, mem, return_attn=False):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1) # [1, 1, 768 global_img_feat_i + 768 masked_token(pose)]
        attention_maps = []
        for blk in self.read_blocks:
            x, _, self_attn, cross_attn = blk(x, mem, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return x[..., -self.v_dim :]
