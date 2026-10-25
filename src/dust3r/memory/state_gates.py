"""Explicit geometry-state gates. Projection names preserve checkpoint loading."""
import torch
from torch import nn

class SpatiallyAwareStateGate(nn.Module):
    def __init__(self, state_dim, img_dim, hidden_dim=256):
        super().__init__()
        # 1. Projection layers: Map State and Image to the same low-dimensional space to compute similarity
        self.q_proj = nn.Linear(state_dim, hidden_dim)
        self.k_proj = nn.Linear(img_dim, hidden_dim)
        self.v_proj = nn.Linear(img_dim, hidden_dim)
        
        # 2. Gate decision network: Makes the update decision based on exclusive local image features
        self.gate_mlp = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialization: Start the gate in a neutral/stable state around 0.5
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, state_feat, img_feat):
        """
        NOTE: img_feat MUST be the complete, un-pooled image feature sequence!
        state_feat: [B, N_state, state_dim] (e.g., [1, 768, 768])
        img_feat:   [B, N_img, img_dim]     (e.g., [1, 576, 1024]) 
        """
        # --- Step 1: Let the State actively "search" for its corresponding local patches in the Image ---
        q = self.q_proj(state_feat) # [B, N_state, H]
        k = self.k_proj(img_feat)   # [B, N_img, H]
        v = self.v_proj(img_feat)   # [B, N_img, H]
        
        # Compute spatial similarity (Attention Matrix) between each State Token and all Image Patches
        scale = q.size(-1) ** -0.5
        attn_logits = torch.bmm(q, k.transpose(1, 2)) * scale # [B, N_state, N_img]
        attn_weights = torch.softmax(attn_logits, dim=-1)     # [B, N_state, N_img]
        
        # --- Step 2: Precisely gather the matching image features ---
        # matched_img_context is no longer a blurry global average; it's the "exclusive view" for each State Token
        matched_img_context = torch.bmm(attn_weights, v)      # [B, N_state, H]
        
        # --- Step 3: Make the comprehensive gating decision ---
        # The State has "seen" its specific local area. Now it can accurately decide whether to update itself.
        x = torch.cat([state_feat, matched_img_context], dim=-1) # [B, N_state, state_dim + H]
        gate = torch.sigmoid(self.gate_mlp(x))                   # [B, N_state, 1]
        return gate

class LearnableStateGate(nn.Module):
    def __init__(self, state_dim, img_dim):
        super().__init__()
        in_features = state_dim + img_dim
        # Projects [Old State + Current Image] down to a bottleneck, then to a scalar
        self.fc1 = nn.Linear(in_features, state_dim // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(state_dim // 2, 1)
        
        # Initialize fc2 to output values near 0 (Sigmoid outputs near 0.5) 
        # so the gate starts neutral and stable
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, state_feat, global_img_feat):
        """
        state_feat: [B, N_state, C] (The historical global state)
        global_img_feat: [B, 1, img_dim] (The current frame's visual summary)
        Returns: [B, N_state, 1] (The soft update gate for each state token)
        """
        B, N, C = state_feat.shape
        # Broadcast the global image feature to every state token
        img_feat_exp = global_img_feat.expand(-1, N, -1)
        
        # Concatenate them so the gate knows "what we remember" and "what we see now"
        x = torch.cat([state_feat, img_feat_exp], dim=-1) # [B, N, state_dim + img_dim]
        
        # Predict the update probability
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(x)))) # [B, N, 1]
        return gate


class ChannelWiseStateGate(nn.Module):
    """Mix each state channel using a broadcast global image summary."""
    def __init__(self, state_dim, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + img_dim, state_dim // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(state_dim // 2, state_dim)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, state_feat, global_img_feat):
        image_context = global_img_feat.expand(-1, state_feat.shape[1], -1)
        joint_features = torch.cat((state_feat, image_context), dim=-1)
        hidden_features = self.act(self.fc1(joint_features))
        return torch.sigmoid(self.fc2(hidden_features))
