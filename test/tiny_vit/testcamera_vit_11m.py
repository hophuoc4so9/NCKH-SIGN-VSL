import cv2
import torch
import numpy as np

# Define DETR Model
import torch
from torch import nn
import timm#
import math

def _get_1d_sincos_pos_embed(length: int, dim: int, temperature: float = 10000.0, device=None):
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
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=1, num_decoder_layers=1, num_queries=25):
        super().__init__()


                # 'tiny_vit_5m_224'  -> Siêu nhẹ
                 # 'tiny_vit_11m_224' -> Cân bằng (Khuyên dùng cho đồ án)
                 # 'tiny_vit_21m_224' -> Mạnh nhất dòng Tiny
        # ResNet-50 backbone
        self.backbone = timm.create_model(
            'tiny_vit_11m_224',
            pretrained=True,
            features_only=True,
            out_indices=(3,)       # Lấy stage cuối cùng
        )

        backbone_out_channels = self.backbone.feature_info[-1]['num_chs']
        # Conversion layer
        self.conv = nn.Conv2d(backbone_out_channels, hidden_dim, 1)



      # 3. Standard Transformer của PyTorch
        self.transformer = nn.Transformer(
            hidden_dim, nheads,
            num_encoder_layers,
            num_decoder_layers,
            batch_first=True,
            dropout=0.1
        )

        # 4. Heads dự đoán
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # 5. Object Queries
        self.num_queries = num_queries
        self.query_pos = nn.Parameter(torch.randn(self.num_queries, hidden_dim))

        self.norm_src = nn.LayerNorm(hidden_dim)
        self.norm_tgt = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        # Trích xuất đặc trưng không gian (Feature Extraction)
        # Output từ backbone là (Batch, Channels, H/32, W/32)
        features = self.backbone(inputs)[-1]

        # Projection (Batch, 256, H/32, W/32)
        feat = self.conv(features)
        bsz, d_model, Hf, Wf = feat.shape

        # Flatten thành chuỗi (Sequence) cho Transformer
        src = feat.flatten(2).permute(0, 2, 1)

        # Positional encoding dựa trên tọa độ thực tế của feature map
        pos = build_2d_sincos_position_embedding(Hf, Wf, d_model, device=feat.device)
        src = self.norm_src(src + pos)

        # Chuẩn bị Query cho Decoder
        tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.norm_tgt(tgt + query_pos)

        # Qua Transformer
        hs = self.transformer(src=src, tgt=tgt)

        return {
            'pred_logits': self.linear_class(hs),
            'pred_boxes': self.linear_bbox(hs).sigmoid()
        }

print("DETR model defined")

CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']

# Define utility functions for boxes and data
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

def box_cxcywh_to_xyxy(x):
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)"""
    x0, y0, w, h = x.unbind(-1)
    b = [(x0 - 0.5 * w), (y0 - 0.5 * h), (x0 + 0.5 * w), (y0 + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """Compute the IoU between boxes"""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    enc_area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (enc_area - union) / enc_area

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

def rescale_bboxes(out_bbox, size):
    """Rescale bounding boxes from normalized coordinates"""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=b.dtype, device=b.device)
    return b

def stacker(batch):
    """Stack batch items"""
    images, targets = [], []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return torch.stack(images), targets

def get_classes():
    return CLASSES

def get_colors():
    return [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

print("Utility functions defined")

# Define Loss Functions
class HungarianMatcher(nn.Module):
    """Hungarian matching for DETR"""
    def __init__(self, weight_dict: dict):
        super().__init__()
        self.class_weighting = weight_dict.get('class_weighting')
        self.bbox_weighting = weight_dict.get('bbox_weighting')
        self.giou_weighting = weight_dict.get('giou_weighting')

    @torch.no_grad()
    def forward(self, yhat, y):
        indices = []
        for batch_idx, target in enumerate(y):
            batch_logits = yhat["pred_logits"][batch_idx]
            batch_boxes = yhat["pred_boxes"][batch_idx]
            batch_prob = batch_logits.softmax(-1)

            tgt_labels = target["labels"].to(torch.long)
            tgt_boxes = target["boxes"].to(batch_boxes.dtype)

            cost_class = -batch_prob[:, tgt_labels]
            cost_bbox = torch.cdist(batch_boxes, tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(batch_boxes),
                box_cxcywh_to_xyxy(tgt_boxes)
            )

            C_batch = (self.bbox_weighting * cost_bbox +
                      self.class_weighting * cost_class +
                      self.giou_weighting * cost_giou).cpu()

            ii, jj = linear_sum_assignment(C_batch)
            indices.append(
                (torch.as_tensor(ii, dtype=torch.int64), torch.as_tensor(jj, dtype=torch.int64))
            )
        return indices

class DETRLoss(nn.Module):
    """DETR Loss computation"""
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def classification_loss(self, yhat, y, indices):
        src_logits = yhat['pred_logits']
        idx = self.get_matched_query_indices(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(y, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def box_loss(self, yhat, y, indices, num_boxes):
        idx = self.get_matched_query_indices(indices)
        src_boxes = yhat['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox_sum = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        loss_giou_sum = loss_giou.sum() / num_boxes

        return {'loss_bbox': loss_bbox_sum, 'loss_giou': loss_giou_sum}

    def get_matched_query_indices(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, yhat, y):
        indices = self.matcher(yhat, y)
        device = next(iter(yhat.values())).device
        y = [{'labels': t['labels'].to(torch.long), 'boxes': t['boxes'].to(torch.float32)} for t in y]

        num_boxes = sum(len(t["labels"]) for t in y)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device).clamp(min=1)

        return {
            'labels': self.classification_loss(yhat, y, indices),
            'boxes': self.box_loss(yhat, y, indices, num_boxes)
        }



CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']

# --- CẤU HÌNH ---
MODEL_PATH = 'D:\\NCKH\\SIGN-VSL\\NCKH-SIGN-VSL\\test\\tiny_vit\\best_signdetr_model_tiny_vit11.pth' # <-- SỬA ĐƯỜNG DẪN NÀY
CONF_THRESH = 0.5  # Ngưỡng độ tin cậy để hiển thị box (có thể chỉnh từ 0.3 - 0.8)
TARGET_SIZE = 224  # Kích thước ảnh đầu vào của model (theo lúc train)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang chạy trên thiết bị: {device}")

num_classes = len(CLASSES)
model = DETR(num_classes=num_classes, hidden_dim=256, nheads=8,
             num_encoder_layers=1, num_decoder_layers=1, num_queries=25)

print("Đang load trọng số model...")
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval() # Chuyển sang chế độ inference, tắt dropout

def preprocess_image(frame):
    # Resize về đúng kích thước model mong đợi (224x224)
    img_resized = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))
    
    # Chuyển BGR (định dạng của OpenCV) sang RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Chuẩn hóa (Normalize) tương tự như lúc train
    img_rgb = img_rgb / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_rgb - mean) / std
    
    # Chuyển thành tensor dạng (Channels, Height, Width)
    img_tensor = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1)
    
    # Thêm batch dimension -> (1, C, H, W) để đưa vào model
    return img_tensor.unsqueeze(0)

# --- 3. MỞ WEBCAM VÀ TEST ---
cap = cv2.VideoCapture(0) # Số 0 thường là camera mặc định của laptop

if not cap.isOpened():
    print("Lỗi: Không thể mở camera! Vui lòng kiểm tra quyền truy cập camera.")
    exit()

print("Bắt đầu nhận diện. Nhấn phím 'q' trên cửa sổ camera để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera.")
        break

    # Lấy kích thước gốc của camera để vẽ box cho chuẩn xác
    h_orig, w_orig, _ = frame.shape

    # Đưa frame vào tiền xử lý
    input_tensor = preprocess_image(frame).to(device)

    # Dự đoán (không tính gradient để tiết kiệm RAM và tăng tốc độ)
    with torch.no_grad():
        outputs = model(input_tensor)

    # Trích xuất kết quả (vì batch_size = 1 nên lấy index 0)
    logits = outputs['pred_logits'][0] # Shape: (25, num_classes + 1)
    boxes = outputs['pred_boxes'][0]   # Shape: (25, 4) dạng [cx, cy, w, h]

    # Tính xác suất bằng Softmax và bỏ class cuối cùng (class đại diện cho Background/Không có gì)
    probs = logits.softmax(-1)[:, :-1]
    max_probs, labels = probs.max(-1)

    # Lọc các bounding box có xác suất lớn hơn ngưỡng cho phép
    keep = max_probs > CONF_THRESH
    
    filtered_boxes = boxes[keep]
    filtered_probs = max_probs[keep]
    filtered_labels = labels[keep]

    # --- 4. HẬU XỬ LÝ VÀ VẼ KẾT QUẢ LÊN FRAME ---
    if len(filtered_boxes) > 0:
        # Chuyển tọa độ box từ [cx, cy, w, h] (chuẩn hóa 0-1) về [x1, y1, x2, y2] tuyệt đối trên frame
        cx, cy, w, h = filtered_boxes.unbind(-1)
        
        x1 = (cx - 0.5 * w) * w_orig
        y1 = (cy - 0.5 * h) * h_orig
        x2 = (cx + 0.5 * w) * w_orig
        y2 = (cy + 0.5 * h) * h_orig
        
        xyxy = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()
        
        for i in range(len(xyxy)):
            box = xyxy[i]
            score = filtered_probs[i].item()
            label_idx = filtered_labels[i].item()
            class_name = CLASSES[label_idx]

            # Ép kiểu tọa độ về số nguyên để OpenCV có thể vẽ
            bx1, by1, bx2, by2 = map(int, box)

            # Vẽ hình chữ nhật (Box màu xanh lá)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            
            # Viết nhãn (Label) và phần trăm độ tin cậy
            text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, text, (bx1, max(by1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow('SignDETR Test - Realtime', frame)

    # Nhấn phím 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp tài nguyên
cap.release()
cv2.destroyAllWindows()