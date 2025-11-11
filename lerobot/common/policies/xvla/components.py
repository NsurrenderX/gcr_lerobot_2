from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import List, Final, Iterable, Tuple

from transformers import AutoProcessor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DATA_DOMAIN_ID = {
    "Bridge": 0,
    "RT1": 1,
    "Calvin": 2,
    "libero": 3,
    "widowx-air": 4,
    "AIR-AGILEX-HQ": 5,
    "robotwin2_abs_ee": 6,
    "robotwin2_clean": 6,
    "robocasa-human": 7,
    "VLABench": 8,
    "AGIBOT-challenge": 9,
    "AIR-AGILEX": 10,
    "AIRBOT": 18,
    
    "cup_full_plus": 19,
    "cup_25_plus": 20,
    "cup_7_plus": 21,
    "cup_100_plus": 22,
    "aloha_bb": 23,
        
    # pretraining
    "robomind-franka": 11,
    "robomind-ur": 12,
    "Droid-Left": 13,
    "Droid-Right": 14,
    "AGIBOT": 15,
    "robomind-agilex": 16,
    "robomind-franka-dual": 17,
}

def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")
    
class EE6DLoss(nn.Module):
    """
    End-effector layout with xyz, 6D rotation, and gripper channels.
    Uses:
      - position (MSE) on xyz pairs
      - rotation-6D (MSE) on two 6D segments
      - gripper (BCE-with-logits) on two gripper indices
    All hyperparameters/indices are hard-coded.
    """

    # ---- Hard-coded hyperparameters/indices ----
    DIM_ACTION: int = 20  # Expected action dimension
    GRIPPER_SCALE: float = 0.1
    GRIPPER_IDX: Tuple[int, int] = (9, 19)

    XYZ_SCALE: float = 100.0
    ROT_SCALE: float = 5.0

    POS_IDX_1: Tuple[int, int, int] = (0, 1, 2)
    POS_IDX_2: Tuple[int, int, int] = (10, 11, 12)

    ROT_IDX_1: Tuple[int, int, int, int, int, int] = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2: Tuple[int, int, int, int, int, int] = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_action: torch.Tensor, target_action: torch.Tensor):
        assert pred_action.shape == target_action.shape, "pred/target shapes must match"
        assert pred_action.dim() == 3, "expected [B, T, D]"
        B, T, D = pred_action.shape  # noqa: F841

        # Validate indices
        _ensure_indices_valid(D, self.GRIPPER_IDX, "gripper_idx")
        _ensure_indices_valid(D, self.POS_IDX_1, "pos_idx_1")
        _ensure_indices_valid(D, self.POS_IDX_2, "pos_idx_2")
        _ensure_indices_valid(D, self.ROT_IDX_1, "rot_idx_1")
        _ensure_indices_valid(D, self.ROT_IDX_2, "rot_idx_2")

        # Gripper BCE (average over both indices)
        g_losses = [
            self.bce(pred_action[:, :, gi], target_action[:, :, gi]) for gi in self.GRIPPER_IDX
        ]
        gripper_loss = sum(g_losses) / len(self.GRIPPER_IDX)
        gripper_loss = gripper_loss * self.GRIPPER_SCALE

        # Position xyz (two triplets)
        pos_loss_1 = self.mse(pred_action[:, :, self.POS_IDX_1], target_action[:, :, self.POS_IDX_1])
        pos_loss_2 = self.mse(pred_action[:, :, self.POS_IDX_2], target_action[:, :, self.POS_IDX_2])
        position_loss = (pos_loss_1 + pos_loss_2) * self.XYZ_SCALE

        # Rotation 6D (two segments)
        rot_loss_1 = self.mse(pred_action[:, :, self.ROT_IDX_1], target_action[:, :, self.ROT_IDX_1])
        rot_loss_2 = self.mse(pred_action[:, :, self.ROT_IDX_2], target_action[:, :, self.ROT_IDX_2])
        rotate6D_loss = (rot_loss_1 + rot_loss_2) * self.ROT_SCALE

        total = position_loss + rotate6D_loss + gripper_loss
        return {
            "position_loss": position_loss,
            "rotate6D_loss": rotate6D_loss,
            "gripper_loss": gripper_loss,
            "total_loss": total,
        }


class JointLoss(nn.Module):
    """
    Joint-space layout with joints + gripper only.
    Uses:
      - joints (MSE) over all non-gripper channels
      - gripper (BCE-with-logits) over two gripper indices
    All hyperparameters/indices are hard-coded.
    """

    # ---- Hard-coded hyperparameters/indices ----
    DIM_ACTION: int = 14  # Expected action dimension
    GRIPPER_SCALE: float = 0.1
    GRIPPER_IDX: Tuple[int, int] = (6, 13)
    JOINTS_SCALE: float = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_action: torch.Tensor, target_action: torch.Tensor):
        assert pred_action.shape == target_action.shape, "pred/target shapes must match"
        assert pred_action.dim() == 3, "expected [B, T, D]"
        B, T, D = pred_action.shape  # noqa: F841

        # Validate
        _ensure_indices_valid(D, self.GRIPPER_IDX, "gripper_idx")

        # Gripper BCE (average over both indices)
        g_losses = [
            self.bce(pred_action[:, :, gi], target_action[:, :, gi]) for gi in self.GRIPPER_IDX
        ]
        gripper_loss = sum(g_losses) / len(self.GRIPPER_IDX)
        gripper_loss = gripper_loss * self.GRIPPER_SCALE

        # Joints = all except grippers
        grip_set = set(self.GRIPPER_IDX)
        joints_idx = tuple(i for i in range(D) if i not in grip_set)
        if len(joints_idx) == 0:
            raise ValueError("No joint indices inferred (D equals number of gripper indices).")

        joints_loss = self.mse(pred_action[:, :, joints_idx], target_action[:, :, joints_idx])
        joints_loss = joints_loss * self.JOINTS_SCALE

        total = joints_loss + gripper_loss
        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
            "total_loss": total,
        }
        

# ------------------------------- Small utils ----------------------------------

def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if we can use PyTorch fused scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")     
 
# ---------------------------------- MLP --------------------------------------

class Mlp(nn.Module):
    """
    MLP used in ViT-style blocks.

    Supports Linear or 1x1 Conv 'linear_layer' for token/channel mixing.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, C] for Linear variant; caller is responsible for shapes.
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -------------------------------- Attention ----------------------------------

class Attention(nn.Module):
    """
    Multi-Head Self-Attention with optional fused SDPA fallback.

    If PyTorch provides `scaled_dot_product_attention`, it will be used
    (usually faster and more stable); otherwise we use a manual implementation.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, C]
            Input sequence.

        Returns
        -------
        Tensor, shape [B, T, C]
            Output sequence after MHSA + projection.
        """
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)  # 3 x [B, H, T, Dh]
        )
        q, k, v = qkv.unbind(0)  # each: [B, H, T, Dh]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )  # [B, H, T, Dh]
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)        # [B, H, T, T]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v                           # [B, H, T, Dh]

        x = x.transpose(1, 2).reshape(B, T, C)     # [B, T, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------- Utilities -----------------------------------

def basic_init(module: nn.Module) -> None:
    """
    Apply a basic initialization scheme to Linear layers.

    - Weight: Xavier uniform initialization.
    - Bias: Set to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
            
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Parameters
    ----------
    t : torch.Tensor
        Shape [B]. Each element is a timestep index, may be fractional.
    dim : int
        Dimensionality of the output embedding.
    max_period : int, default=100
        Controls the minimum frequency of the sinusoids.

    Returns
    -------
    torch.Tensor
        Shape [B, dim]. Sinusoidal embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# ------------------------------- Core Layers ----------------------------------

class DomainAwareLinear(nn.Module):
    """
    Linear layer with domain-conditioned parameters (per-sample).

    Each domain has its own weight and bias vectors, stored in embeddings.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 20) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [B, I] or [B, T, I]
        domain_id : LongTensor
            [B], domain indices.

        Returns
        -------
        Tensor
            [B, O] or [B, T, O]
        """
        B = domain_id.shape[0]
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, self.output_size)
        y = torch.matmul(x, W) + b.view(B, 1, self.output_size)
        if squeeze_T:
            y = y.squeeze(1)
        return y


class TransformerBlock(nn.Module):
    """
    Standard Transformer block (pre-LN): LN → MHSA → residual, LN → MLP → residual.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, [B, T, H]

        Returns
        -------
        Tensor, [B, T, H]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
# --------------------------- Main Model ---------------------------------------

class SoftPromptedTransformer(nn.Module):
    """
    Multi-modal, domain-aware Transformer with optional soft prompts.

    See parameter and forward I/O descriptions inside the docstrings.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        multi_model_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(multi_model_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(multi_model_input_size, hidden_size, num_domains=num_domains)
        else:
            self.vlm_proj = nn.Linear(multi_model_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_model_input_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(
            dim_action + dim_time + dim_propio, hidden_size, num_domains=num_domains
        )
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)

        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)

        self.apply(basic_init)

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Inputs
        ------
        domain_id : [B]
        vlm_features : [B, T_vlm, D]
        aux_visual_inputs : [B, T_aux, D]
        action_with_noise : [B, T_action, dim_action]
        proprio : [B, dim_propio]
        t : [B]

        Returns
        -------
        Tensor
            Predicted actions, [B, T_action, dim_action]
        """
        B, num_actions = action_with_noise.shape[:2]

        # Encode (action + proprio + time) → tokens
        time_emb = timestep_embedding(t, self.dim_time)                     # [B, dim_time]
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens, domain_id)                   # [B, T_action, H]

        # Project visual streams and concatenate
        if self.use_hetero_proj:
            x = torch.cat(
                [x, self.vlm_proj(vlm_features, domain_id), self.aux_visual_proj(aux_visual_inputs, domain_id)],
                dim=1,
            )
        else:
            x = torch.cat([x, self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual_inputs)], dim=1)

        # Add positional embeddings (truncate if needed)
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
            )
        x = x + self.pos_emb[:, :seq_len, :]

        # Append soft prompts
        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(B, self.len_soft_prompts, self.hidden_size)
            x = torch.cat([x, soft_prompts], dim=1)

        # Transformer backbone
        for block in self.blocks:
            x = block(x)

        # Decode only the action segment
        return self.action_decoder(self.norm(x[:, :num_actions]), domain_id)
    
# ----------------------------- Language Preprocessor --------------------------

class LanguagePreprocessor:
    """
    A lightweight wrapper for tokenizing natural language instructions.

    Parameters
    ----------
    encoder_name : str
        Hugging Face model name or local path (e.g. "bert-base-uncased").
    device : str, default="cuda"
        Device to move the tokenized tensors to ("cuda" or "cpu").
    max_length : int, default=50
        Maximum sequence length for tokenization.

    Methods
    -------
    encode_language(language_instruction: List[str]) -> Dict[str, torch.Tensor]
        Tokenizes a batch of instructions into input IDs.
    """

    def __init__(self, encoder_name: str, max_length: int = 50):
        self.preprocessor = AutoProcessor.from_pretrained(encoder_name, trust_remote_code=True)
        self.max_length = max_length

    @torch.no_grad()
    def encode_language(self, language_instruction: List[str]):
        """
        Tokenize a list of language instructions.

        Parameters
        ----------
        language_instruction : List[str]
            List of natural language commands/instructions.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with:
            - "input_ids": tensor of shape [B, max_length], on self.device.
        """
        if isinstance(language_instruction, str): language_instruction = [language_instruction]
        inputs = self.preprocessor.tokenizer(
            language_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        return {"input_ids": inputs["input_ids"]}


# ----------------------------- Image Preprocessor -----------------------------


class ImagePreprocessor:
    """
    Prepares a fixed number of image views with normalization.

    Parameters
    ----------
    num_views : int, default=3
        Number of image views expected. If fewer are provided, zero-padded
        placeholders are added.

    Methods
    -------
    __call__(images: List[Image]) -> Dict[str, torch.Tensor]
        Applies preprocessing to a batch of images.
    """

    def __init__(self, num_views: int = 3):
        self.num_views = num_views
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                inplace=True,
            ),
        ])

    def __call__(self, images):
        """
        Preprocess and pad a list of images.

        Parameters
        ----------
        images : List[PIL.Image]
            List of input images.

        Returns
        -------
        Dict[str, torch.Tensor]
            - "image_input": tensor of shape [1, num_views, 3, 224, 224].
              If fewer than num_views are provided, zero-padding is applied.
            - "image_mask": tensor of shape [1, num_views], bool type,
              indicating which slots correspond to valid images.
        """
        # Apply transforms to each provided image
        x = torch.stack([self.image_transform(img) for img in images])

        # Pad with zero-images if fewer than num_views are given
        V_exist = x.size(0)
        if V_exist < self.num_views:
            x = torch.cat(
                [x, x.new_zeros(self.num_views - V_exist, *x.shape[1:])],
                dim=0,
            )

        # Build image mask: True for valid slots, False for padded ones
        image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=x.device)
        image_mask[:V_exist] = True

        return {
            "image_input": x.unsqueeze(0),   # [1, num_views, 3, 224, 224]
            "image_mask": image_mask.unsqueeze(0),  # [1, num_views]
        }
        
def find_domain_id(domain_name):
    if domain_name in DATA_DOMAIN_ID:
        return DATA_DOMAIN_ID[domain_name]
    else:
        return 30

# -----------------------
# Base Utils
# -----------------------
def _as_tensor(x, dtype=None, device=None):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype, device=device)
    return x

# -----------------------
# 6D <-> matrix
# -----------------------
def ortho6d_to_matrix(ortho6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    ortho6d: (..., 6) where first 3 are column0, next 3 are column1
    returns: (..., 3, 3) with columns [x, y, z]
    """
    ortho6d = _as_tensor(ortho6d)
    if ortho6d.shape[-1] != 6:
        raise ValueError("last dim must be 6")
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]

    x = F.normalize(x_raw, p=2, dim=-1, eps=eps)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, p=2, dim=-1, eps=eps)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-1)  # (...,3,3) columns are x,y,z
    return R

def matrix_to_ortho6d(R: torch.Tensor) -> torch.Tensor:
    """
    R: (..., 3, 3)
    returns: (..., 6) with concat of first two columns [col0, col1]
    """
    R = _as_tensor(R)
    if R.shape[-2:] != (3,3):
        raise ValueError("R must have shape (...,3,3)")
    col0 = R[..., :, 0]  # (...,3)
    col1 = R[..., :, 1]
    return torch.cat([col0, col1], dim=-1)  # (...,6)

# -----------------------
# Quaternion <-> matrix
# -----------------------
def quaternion_to_matrix(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    q: (...,4) in (w, x, y, z) format
    returns: (...,3,3)
    """
    q = _as_tensor(q)
    if q.shape[-1] != 4:
        raise ValueError("Quaternion must have last dim = 4 (w,x,y,z)")

    q = q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))  # normalize
    w, x, y, z = q.unbind(dim=-1)  # each (...,)

    # compute matrix elements
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R00 = ww + xx - yy - zz
    R01 = 2 * (xy - wz)
    R02 = 2 * (xz + wy)

    R10 = 2 * (xy + wz)
    R11 = ww - xx + yy - zz
    R12 = 2 * (yz - wx)

    R20 = 2 * (xz - wy)
    R21 = 2 * (yz + wx)
    R22 = ww - xx - yy + zz

    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1),
    ], dim=-2)  # (...,3,3)
    return R

def matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    R: (...,3,3)
    returns: (...,4) as (w,x,y,z)
    Robust numerically using standard branch on trace.
    """
    R = _as_tensor(R)
    if R.shape[-2:] != (3,3):
        raise ValueError("R must have shape (...,3,3)")

    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    trace = m00 + m11 + m22

    # prepare containers
    w = torch.empty_like(trace)
    x = torch.empty_like(trace)
    y = torch.empty_like(trace)
    z = torch.empty_like(trace)

    # case trace > 0
    pos = trace > 0
    if pos.any():
        s = torch.sqrt(trace[pos] + 1.0) * 2.0  # s = 4*w
        w[pos] = 0.25 * s
        x[pos] = (R[pos][...,2,1] - R[pos][...,1,2]) / s
        y[pos] = (R[pos][...,0,2] - R[pos][...,2,0]) / s
        z[pos] = (R[pos][...,1,0] - R[pos][...,0,1]) / s

    # other cases: determine which diagonal is biggest
    not_pos = ~pos
    if not_pos.any():
        # create view of sub-block
        m00_n = m00[not_pos]
        m11_n = m11[not_pos]
        m22_n = m22[not_pos]
        Rn = R[not_pos]

        cond_x = (m00_n > m11_n) & (m00_n > m22_n)
        cond_y = (m11_n > m22_n) & (~cond_x)
        cond_z = ~(cond_x | cond_y)

        # X biggest
        if cond_x.any():
            s = torch.sqrt(1.0 + m00_n[cond_x] - m11_n[cond_x] - m22_n[cond_x]) * 2.0
            x[not_pos][cond_x] = 0.25 * s
            w[not_pos][cond_x] = (Rn[cond_x][...,2,1] - Rn[cond_x][...,1,2]) / s
            y[not_pos][cond_x] = (Rn[cond_x][...,0,1] + Rn[cond_x][...,1,0]) / s
            z[not_pos][cond_x] = (Rn[cond_x][...,0,2] + Rn[cond_x][...,2,0]) / s

        # Y biggest
        if cond_y.any():
            s = torch.sqrt(1.0 + m11_n[cond_y] - m00_n[cond_y] - m22_n[cond_y]) * 2.0
            y[not_pos][cond_y] = 0.25 * s
            w[not_pos][cond_y] = (Rn[cond_y][...,0,2] - Rn[cond_y][...,2,0]) / s
            x[not_pos][cond_y] = (Rn[cond_y][...,0,1] + Rn[cond_y][...,1,0]) / s
            z[not_pos][cond_y] = (Rn[cond_y][...,1,2] + Rn[cond_y][...,2,1]) / s

        # Z biggest
        if cond_z.any():
            s = torch.sqrt(1.0 + m22_n[cond_z] - m00_n[cond_z] - m11_n[cond_z]) * 2.0
            z[not_pos][cond_z] = 0.25 * s
            w[not_pos][cond_z] = (Rn[cond_z][...,1,0] - Rn[cond_z][...,0,1]) / s
            x[not_pos][cond_z] = (Rn[cond_z][...,0,2] + Rn[cond_z][...,2,0]) / s
            y[not_pos][cond_z] = (Rn[cond_z][...,1,2] + Rn[cond_z][...,2,1]) / s

    q = torch.stack([w, x, y, z], dim=-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return q

# -----------------------
# Euler(zyx) <-> matrix
# -----------------------
def euler_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """
    euler: (..., 3) angles (roll_x, pitch_y, yaw_z) ????
    NOTE: We assume input order is (roll, pitch, yaw) but the ZYX composition
    corresponds to R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    For clarity, the input vector is [roll, pitch, yaw] in radians.
    returns: (...,3,3)
    """
    euler = _as_tensor(euler)
    if euler.shape[-1] != 3:
        raise ValueError("euler must have last dim 3: (roll, pitch, yaw)")

    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    cr = torch.cos(roll); sr = torch.sin(roll)
    cp = torch.cos(pitch); sp = torch.sin(pitch)
    cz = torch.cos(yaw); sz = torch.sin(yaw)

    # Compose R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R00 = cz * cp
    R01 = cz * sp * sr - sz * cr
    R02 = cz * sp * cr + sz * sr

    R10 = sz * cp
    R11 = sz * sp * sr + cz * cr
    R12 = sz * sp * cr - cz * sr

    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr

    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1),
    ], dim=-2)
    return R

def matrix_to_euler(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    R: (...,3,3)
    returns: (...,3) = (roll, pitch, yaw) such that R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Handles gimbal lock approximately: when cos(pitch) ~ 0, yaw set to 0 and roll derived.
    """
    R = _as_tensor(R)
    if R.shape[-2:] != (3,3):
        raise ValueError("R must have shape (...,3,3)")

    R20 = R[..., 2, 0]
    # clamp for numerical safety
    pitch = torch.asin((-R20).clamp(-1.0, 1.0))

    cos_pitch = torch.cos(pitch)
    singular = cos_pitch.abs() < eps

    # default (non-singular) formulas
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    yaw  = torch.atan2(R[..., 1, 0], R[..., 0, 0])

    # handle singularities: when |cos(pitch)| ~ 0
    if singular.any():
        # for those entries, set yaw = 0 and compute roll from first row instead
        idx = singular
        # When pitch ~ +pi/2 or -pi/2, R20 = -sin(pitch) -> check sign
        # Use alternative computations to avoid division by zero.
        # Set yaw = 0, roll = atan2(-R01, R11)  (common fallback)
        roll_sing = torch.atan2(-R[..., 0, 1], R[..., 1, 1])
        roll = torch.where(idx, roll_sing, roll)
        yaw = torch.where(idx, torch.zeros_like(yaw), yaw)

    return torch.stack([roll, pitch, yaw], dim=-1)

# -----------------------
# Convenience wrappers
# -----------------------
def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    return matrix_to_quaternion(euler_to_matrix(euler))

def quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
    return matrix_to_euler(quaternion_to_matrix(q))

def ortho6d_to_quaternion(ortho6d: torch.Tensor) -> torch.Tensor:
    return matrix_to_quaternion(ortho6d_to_matrix(ortho6d))

def quaternion_to_ortho6d(q: torch.Tensor) -> torch.Tensor:
    return matrix_to_ortho6d(quaternion_to_matrix(q))

def euler_to_ortho6d(euler: torch.Tensor) -> torch.Tensor:
    return matrix_to_ortho6d(euler_to_matrix(euler))

def ortho6d_to_euler(ortho6d: torch.Tensor) -> torch.Tensor:
    return matrix_to_euler(ortho6d_to_matrix(ortho6d))