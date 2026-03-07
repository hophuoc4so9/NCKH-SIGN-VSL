import torch
from torch import nn
from torchvision.models import swin_t, Swin_T_Weights
import sys 
from colorama import Fore 
from utils.logger import get_logger
from utils.rich_handlers import ModelHandler
from torchinfo import summary
import sys 
import math


def _get_1d_sincos_pos_embed(length: int, dim: int, temperature: float = 10000.0, device=None):
    assert dim % 2 == 0
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # (L,1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(temperature) / dim)
    )  # (dim/2)
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, dim)


def build_2d_sincos_position_embedding(height: int, width: int, dim: int, device=None):
    """Create 2D sine-cos positional encoding of shape (1, H*W, dim).
    Half dims for Y, half for X.
    """
    assert dim % 2 == 0, "positional dim must be even"
    dim_half = dim // 2
    pe_y = _get_1d_sincos_pos_embed(height, dim_half, device=device)  # (H, dim/2)
    pe_x = _get_1d_sincos_pos_embed(width, dim_half, device=device)   # (W, dim/2)
    # Combine to (H, W, dim)
    pos = torch.zeros(height, width, dim, device=device, dtype=torch.float32)
    pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, width, -1)
    pos[:, :, dim_half:] = pe_x[None, :, :].expand(height, -1, -1)
    pos = pos.view(1, height * width, dim)  # (1, H*W, dim)
    return pos


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=1, num_decoder_layers=1, num_queries=25):
        super().__init__()
        
        # Initialize logger and model handler
        self.logger = get_logger("model")
        self.model_handler = ModelHandler()
        
        # Log model configuration
        model_config = {
            "Model Type": "DETR (Detection Transformer)",
            "Number of Classes": num_classes,
            "Hidden Dimension": hidden_dim,
            "Attention Heads": nheads,
            "Encoder Layers": num_encoder_layers,
            "Decoder Layers": num_decoder_layers,
            "Object Queries": num_queries,
            "Backbone": "Swin-T (ImageNet pretrained)"
        }
        self.model_handler.log_model_architecture(model_config)

        # Swin-T backbone (output channels = 768)
        swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.backbone = swin.features  # Chỉ lấy phần features
        self.backbone_norm = swin.norm  # LayerNorm sau features

        # Swin-T output là 768 channels, không phải 2048 như ResNet
        self.conv = nn.Conv2d(768, hidden_dim, 1)

        # Transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers,
            batch_first=True, dropout=0.1)

        # Prediction heads
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # Query embeddings
        self.num_queries = num_queries
        self.query_pos = nn.Parameter(torch.randn(self.num_queries, hidden_dim))

        # Normalizations
        self.norm_src = nn.LayerNorm(hidden_dim)
        self.norm_tgt = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        # Output shape: (B, H/32, W/32, 768)
        x = self.backbone(inputs)  # (B, H/32, W/32, 768)
        x = self.backbone_norm(x)  # LayerNorm

        # Chuyển từ (B, H, W, C) sang (B, C, H, W) để dùng Conv2d
        x = x.permute(0, 3, 1, 2)  # (B, 768, H/32, W/32)

        feat = self.conv(x)  # (B, hidden_dim, Hf, Wf)
        bsz, d_model, Hf, Wf = feat.shape
        src = feat.flatten(2).permute(0, 2, 1)  # (B, Hf*Wf, hidden_dim)

        pos = build_2d_sincos_position_embedding(Hf, Wf, d_model, device=feat.device)
        src = self.norm_src(src + pos)

        tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.norm_tgt(tgt + query_pos)

        hs = self.transformer(src=src, tgt=tgt)

        return {
            'pred_logits': self.linear_class(hs),
            'pred_boxes': self.linear_bbox(hs).sigmoid()
        }
    
    def log_model_info(self):
        """Log model parameter information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.model_handler.log_parameters_count(total_params, trainable_params)
        
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights with logging."""
        try:
            self.load_state_dict(torch.load(checkpoint_path))
            self.model_handler.log_model_loading(checkpoint_path, success=True)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            self.model_handler.log_model_loading(checkpoint_path, success=False)


if __name__ == '__main__': 
    model = DETR(num_classes=3)
    summary(model, (5,3,224,224))