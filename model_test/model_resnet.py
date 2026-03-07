"""
DETR Model với ResNet50 backbone cho Sign Language Detection
"""
import math
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


def _get_1d_sincos_pos_embed(length: int, dim: int, temperature: float = 10000.0, device=None):
    """Tạo 1D sinusoidal positional embedding"""
    assert dim % 2 == 0
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(temperature) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def build_2d_sincos_position_embedding(height: int, width: int, dim: int, device=None):
    """Tạo 2D sinusoidal positional embedding"""
    assert dim % 2 == 0, "positional dim must be even"
    dim_half = dim // 2
    pe_y = _get_1d_sincos_pos_embed(height, dim_half, device=device)
    pe_x = _get_1d_sincos_pos_embed(width, dim_half, device=device)
    pos = torch.zeros(height, width, dim, device=device, dtype=torch.float32)
    pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, width, -1)
    pos[:, :, dim_half:] = pe_x[None, :, :].expand(height, -1, -1)
    pos = pos.view(1, height * width, dim)
    return pos


class DETR(nn.Module):
    """
    DETR (DEtection TRansformer) model với ResNet50 backbone
    cho bài toán Sign Language Detection
    """
    def __init__(self, num_classes=22, hidden_dim=256, nheads=8,
                 num_encoder_layers=1, num_decoder_layers=1, num_queries=25):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # ResNet-50 backbone (pretrained trên ImageNet)
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        # Conversion layer: giảm channels từ 2048 xuống hidden_dim
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
            dropout=0.1
        )

        # Prediction heads
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)  # +1 cho class "no object"
        self.linear_bbox = nn.Linear(hidden_dim, 4)  # (cx, cy, w, h)

        # Query embeddings (learnable)
        self.query_pos = nn.Parameter(torch.randn(self.num_queries, hidden_dim))

        # Layer normalizations
        self.norm_src = nn.LayerNorm(hidden_dim)
        self.norm_tgt = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        """
        Forward pass
        Args:
            inputs: Tensor shape (B, 3, H, W) - batch of images
        Returns:
            dict với 'pred_logits' và 'pred_boxes'
        """
        # Backbone feature extraction
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Project to hidden dimension
        feat = self.conv(x)
        bsz, d_model, Hf, Wf = feat.shape
        
        # Flatten spatial dimensions
        src = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)

        # Add positional embedding
        pos = build_2d_sincos_position_embedding(Hf, Wf, d_model, device=feat.device)
        src = self.norm_src(src + pos)

        # Prepare target (object queries)
        tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.norm_tgt(tgt + query_pos)

        # Transformer
        hs = self.transformer(src=src, tgt=tgt)

        # Prediction heads
        return {
            'pred_logits': self.linear_class(hs),
            'pred_boxes': self.linear_bbox(hs).sigmoid()
        }


# Danh sách classes
CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']


def box_cxcywh_to_xyxy(x):
    """Convert boxes từ (cx, cy, w, h) sang (x1, y1, x2, y2)"""
    x0, y0, w, h = x.unbind(-1)
    b = [(x0 - 0.5 * w), (y0 - 0.5 * h), (x0 + 0.5 * w), (y0 + 0.5 * h)]
    return torch.stack(b, dim=-1)


def rescale_bboxes(out_bbox, size):
    """Rescale bounding boxes từ normalized coordinates về pixel coordinates"""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=b.dtype, device=b.device)
    return b


if __name__ == "__main__":
    # Test model
    model = DETR(num_classes=22)
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"pred_logits shape: {output['pred_logits'].shape}")
    print(f"pred_boxes shape: {output['pred_boxes'].shape}")
