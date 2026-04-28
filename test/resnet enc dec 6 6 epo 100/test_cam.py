import os, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet50, ResNet50_Weights

# ── Cấu hình ──────────────────────────────────────────────────
CLASSES = ['A','B','C','D','E','G','H','I','K','L',
           'M','N','O','P','Q','R','S','T','U','V','X','Y']
NUM_CLASSES  = len(CLASSES)
HIDDEN_DIM   = 256
NHEADS       = 8

NUM_QUERIES  = 10
DIM_FF       = 1024
DROPOUT      = 0.0
CONF_THR     = 0.35
MODEL_PATH   = "D:\\NCKH\\SIGN-VSL\\NCKH-SIGN-VSL\\test\\v1\\best_signdetr_model (3).pth"  
IMG_SIZE     = 512

NUM_EPOCHS      = 100
LR_BACKBONE     = 1e-5
LR_TRANSFORMER  = 1e-4
WEIGHT_DECAY    = 1e-4
WARMUP_EPOCHS   = 10
EMA_START_EPOCH = 10
HIDDEN_DIM      = 256
NHEADS          = 8
ENC_LAYERS      = 6
DEC_LAYERS      = 6
NUM_QUERIES     = 25
DIM_FF          = 2048
DROPOUT         = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Định nghĩa model (copy y chang notebook) ──────────────────

def _get_1d_sincos_pos_embed(length, dim, temperature=10000.0, device=None):
    assert dim % 2 == 0
    pos  = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div  = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                     * (-math.log(temperature) / dim))
    pe   = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

def build_2d_sincos_pos_embed(H, W, dim, device=None):
    assert dim % 2 == 0
    d = dim // 2
    pe_y = _get_1d_sincos_pos_embed(H, d, device=device)  # (H, d)
    pe_x = _get_1d_sincos_pos_embed(W, d, device=device)  # (W, d)
    pos  = torch.zeros(H, W, dim, device=device)
    pos[:, :, :d] = pe_y[:, None, :].expand(-1, W, -1)
    pos[:, :, d:] = pe_x[None, :, :].expand(H, -1, -1)
    return pos.view(1, H * W, dim)                         # (1, HW, dim)

# 5. CUSTOM TRANSFORMER DECODER WITH INTERMEDIATE OUTPUTS
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn   = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff1          = nn.Linear(d_model, dim_feedforward)
        self.ff2          = nn.Linear(dim_feedforward, d_model)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.norm3       = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)
        self.act         = nn.GELU()

    def forward(self, tgt, memory, query_pos, pos):
        # self-attention on queries
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt  = self.norm1(tgt + self.dropout(tgt2))
        # cross-attention to encoder memory
        tgt2, _ = self.cross_attn(tgt + query_pos, memory + pos, memory)
        tgt  = self.norm2(tgt + self.dropout(tgt2))
        # feed-forward
        tgt2 = self.ff2(self.dropout(self.act(self.ff1(tgt))))
        tgt  = self.norm3(tgt + self.dropout(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    """Returns a list of intermediate outputs (one per layer)."""
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(decoder_layer.ff2.out_features)

    def forward(self, tgt, memory, query_pos, pos):
        out, intermediates = tgt, []
        for layer in self.layers:
            out = layer(out, memory, query_pos, pos)
            intermediates.append(self.norm(out))
        return torch.stack(intermediates)               # (num_layers, B, Q, d)
# 6. DETR MODEL
from torchvision.models import resnet50, ResNet50_Weights

class DETR(nn.Module):
    def __init__(self,
                 num_classes,
                 hidden_dim       = 256,
                 nheads           = 8,
                 num_encoder_layers = 6,   # default: DETR-paper setting
                 num_decoder_layers = 6,
                 num_queries      = 25,
                 dim_feedforward  = 2048,
                 dropout          = 0.1):
        super().__init__()

        # ── Backbone
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # keep only feature extractor, drop avg-pool + fc
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

        # ── Projection: 2048 → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Conv2d(2048, hidden_dim, 1),
            nn.GroupNorm(32, hidden_dim)               # more stable than BN under small batches
        )

        # ── Transformer encoder (standard PyTorch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True)                           # Pre-LN: faster convergence
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers,
                                              norm=nn.LayerNorm(hidden_dim))

        # ── Transformer decoder (custom, returns intermediates) ─
        dec_layer = TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers)

        # ── Prediction heads (shared across layers via FFN) ────
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

        # ── Learnable query embeddings ─────────────────────────
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)   # [Q, d]

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj[0].weight)
        nn.init.constant_(self.input_proj[0].bias, 0)
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0)
        for m in self.bbox_embed:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ── 1. Backbone features ───────────────────────────────
        feat = self.backbone(x)                        # (B, 2048, H/32, W/32)
        feat = self.input_proj(feat)                   # (B, d, Hf, Wf)
        B, d, Hf, Wf = feat.shape

        # ── 2. Flatten + positional embedding ─────────────────
        src = feat.flatten(2).permute(0, 2, 1)         # (B, HW, d)
        pos = build_2d_sincos_pos_embed(Hf, Wf, d, device=feat.device)  # (1, HW, d)

        # ── 3. Encoder ─────────────────────────────────────────
        memory = self.encoder(src + pos)               # (B, HW, d)

        # ── 4. Decoder ─────────────────────────────────────────
        tgt        = torch.zeros(B, self.num_queries, d, device=x.device)
        query_pos  = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d)

        # hs: (num_layers, B, Q, d)
        hs = self.decoder(tgt, memory, query_pos, pos)

        # ── 5. Prediction heads applied to ALL decoder layers ──
        outputs_class = self.class_embed(hs)           # (L, B, Q, num_cls+1)
        outputs_coord = self.bbox_embed(hs).sigmoid()  # (L, B, Q, 4)

        # Primary output = last decoder layer
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes':  outputs_coord[-1],
        }

        # Auxiliary outputs (all other layers) for aux loss
        out['aux_outputs'] = [
            {'pred_logits': outputs_class[i], 'pred_boxes': outputs_coord[i]}
            for i in range(len(self.decoder.layers) - 1)
        ]
        return out

    


# ── Load model ─────────────────────────────────────────────────
model = DETR(
    num_classes        = NUM_CLASSES,
    hidden_dim         = HIDDEN_DIM,
    nheads             = NHEADS,
    num_encoder_layers = ENC_LAYERS,
    num_decoder_layers = DEC_LAYERS,
    num_queries        = NUM_QUERIES,
    dim_feedforward    = DIM_FF,
    dropout            = DROPOUT,
).to(device)

ckpt  = torch.load(MODEL_PATH, map_location=device)
state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
print(f"✅ Load model thành công!")
print(f"   Epoch: {ckpt.get('epoch', 'N/A')}  |  Loss: {ckpt.get('loss', 0):.4f}")
# ── Transform (không dùng albumentations trong vòng lặp camera
#    vì chậm — dùng OpenCV thuần cho real-time) ─────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR frame → (1,3,512,512) tensor trên device"""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

# Màu cho từng class (22 class → bảng màu cố định)
np.random.seed(42)
CLASS_COLORS = {
    cls: tuple(int(c) for c in np.random.randint(100, 255, 3))
    for cls in CLASSES
}

# ── Vòng lặp camera real-time ──────────────────────────────────
# Phím điều khiển:
#   Q hoặc ESC  → thoát
#   +/-         → tăng/giảm ngưỡng confidence
#   S           → chụp lưu ảnh hiện tại

CAMERA_ID   = 0          # 0 = webcam mặc định, đổi sang 1,2... nếu có nhiều camera
SKIP_FRAMES = 2          # chỉ inference mỗi N frame → giữ FPS cao
WINDOW_NAME = "SignDETR — VSL Real-time  |  Q: quit  |  +/-: conf  |  S: save"

def draw_predictions(frame, results, conf_thr):
    """Vẽ bbox + label lên frame BGR"""
    h, w = frame.shape[:2]
    for r in results:
        cls_name = r['class']
        conf     = r['conf']
        cx, cy, bw, bh = r['box']

        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = CLASS_COLORS[cls_name]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label nền + chữ
        label = f"{cls_name}  {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def run_camera(camera_id=CAMERA_ID, conf_thr=CONF_THR):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Không mở được camera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"✅ Camera {camera_id} đã mở  — {WINDOW_NAME}")
    print(f"   Conf threshold: {conf_thr:.2f}")

    frame_count = 0
    last_results = []
    fps_time = cv2.getTickCount()
    fps_display = 0.0
    save_count  = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không đọc được frame từ camera")
                break

            frame_count += 1

            # Inference mỗi SKIP_FRAMES frame
            if frame_count % SKIP_FRAMES == 0:
                inp = preprocess(frame)
                out = model(inp)

                logits = out['pred_logits'][0]
                boxes  = out['pred_boxes'][0]
                probs  = logits.softmax(-1)[:, :-1]
                scores, cls_ids = probs.max(-1)

                keep = scores >= conf_thr
                last_results = []
                for score, cls_id, box in zip(scores[keep], cls_ids[keep], boxes[keep]):
                    last_results.append({
                        'class': CLASSES[cls_id.item()],
                        'conf' : round(score.item(), 4),
                        'box'  : box.cpu().tolist()
                    })
                last_results.sort(key=lambda x: x['conf'], reverse=True)

            # Vẽ kết quả
            display = frame.copy()
            draw_predictions(display, last_results, conf_thr)

            # FPS
            if frame_count % 15 == 0:
                t_now = cv2.getTickCount()
                fps_display = 15 * cv2.getTickFrequency() / (t_now - fps_time)
                fps_time = t_now

            # HUD: FPS + conf threshold
            cv2.putText(display, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(display, f"Conf: {conf_thr:.2f}  (+/-)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if not last_results:
                cv2.putText(display, "Khong phat hien...", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

            cv2.imshow(WINDOW_NAME, display)

            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):        # Q / ESC → thoát
                print("👋 Thoát camera")
                break
            elif key in (ord('+'), ord('=')):           # tăng conf
                conf_thr = min(conf_thr + 0.05, 0.95)
                print(f"   Conf threshold → {conf_thr:.2f}")
            elif key == ord('-'):                       # giảm conf
                conf_thr = max(conf_thr - 0.05, 0.05)
                print(f"   Conf threshold → {conf_thr:.2f}")
            elif key in (ord('s'), ord('S')):           # lưu ảnh
                save_count += 1
                fname = f"capture_{save_count:03d}.jpg"
                cv2.imwrite(fname, display)
                print(f"📸 Đã lưu: {fname}")

    cap.release()
    cv2.destroyAllWindows()

# ── Chạy ──────────────────────────────────────────────────────
run_camera(camera_id=0, conf_thr=CONF_THR)