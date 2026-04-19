"""
Test Sign Language Detection bằng Camera
Sử dụng DETR Model backbone Swin-T

Cách sử dụng:
    python test_camera_swin_t.py

Controls:
    q - Thoát
    s - Lưu ảnh
    +/- - Tăng/giảm threshold
"""
import cv2
import numpy as np
import torch
import time
import os
import colorsys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import model
from model_swin_t import DETR

# MODEL_PATH = "E:\\git\\SignDETR\\model_test\\swin_t\\best_signdetr_model.pth"
# MODEL_PATH = "E:\\git\\SignDETR\\model_test\\swin_t\\checkpoint_epoch_100.pth"

# MODEL_PATH = "E:\\git\\SignDETR\\model_test\\swin_t_v2\\best_signdetr_model (1).pth"
MODEL_PATH = "E:\\git\\SignDETR\\model_test\\swin_t_v2\\checkpoint_epoch_100 (1).pth"

CONFIDENCE_THRESHOLD = 0.5
NUM_CLASSES = 22
CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']

def get_colors(num_classes=22):
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors

def predict_full_frame(model, frame, transforms, device, threshold=0.3):
    start_time = time.time()
    height, width = frame.shape[:2]
    with torch.no_grad():
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = transforms(image=rgb_image)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        result = model(input_tensor)
        probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]
        max_probs, max_classes = probabilities.max(-1)
        keep_mask = max_probs[0] > threshold
        detections = []
        for i in range(keep_mask.shape[0]):
            if keep_mask[i]:
                cls_idx = max_classes[0, i].item()
                conf = max_probs[0, i].item()
                bbox = result['pred_boxes'][0, i]
                cx, cy, w, h = bbox.cpu().numpy()
                x1 = int((cx - w/2) * width)
                y1 = int((cy - h/2) * height)
                x2 = int((cx + w/2) * width)
                y2 = int((cy + h/2) * height)
                detections.append({
                    'class': CLASSES[cls_idx],
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
    inference_time = (time.time() - start_time) * 1000
    return detections, inference_time

def main():
    print("=" * 60)
    print("   SIGN LANGUAGE DETECTION - Camera Test (Swin-T)")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n📂 Loading model từ: {MODEL_PATH}")
    model = DETR(num_classes=NUM_CLASSES)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = None
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
            if 'loss' in checkpoint:
                print(f"   Loss: {checkpoint.get('loss'):.4f}")
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Sửa key nếu có prefix 'backbone.features.'
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.features.'):
                new_k = k.replace('backbone.features.', 'backbone.', 1)
            elif k.startswith('backbone_norm.'):
                new_k = k.replace('backbone_norm.', 'backbone_norm.', 1)
            else:
                new_k = k
            new_state_dict[new_k] = v
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print(" Load model thành công!")
        except Exception as e2:
            print(f"❌ Lỗi load model (đã sửa key): {e2}")
            return
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        print("   Hãy chắc chắn file model tồn tại và đúng format.")
        return
    model.to(device)
    model.eval()
    transforms = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    print("\n📷 Đang mở webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Camera: {frame_width}x{frame_height}")
    colors = get_colors(NUM_CLASSES)
    print("\n" + "=" * 60)
    print("   CONTROLS:")
    print("   q - Thoát")
    print("   s - Lưu screenshot")
    print("   +/- - Tăng/giảm confidence threshold")
    print("=" * 60 + "\n")
    frame_count = 0
    confidence_threshold = CONFIDENCE_THRESHOLD
    fps_start_time = time.time()
    fps = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được frame!")
            break
        frame_count += 1
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        detections, inference_time = predict_full_frame(
            model, frame, transforms, device, confidence_threshold
        )
        detected_signs = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            conf = det['confidence']
            class_idx = CLASSES.index(class_name)
            color = colors[class_idx]
            detected_signs.append((class_name, conf))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display, (x1, y1-text_h-10), (x1+text_w+10, y1), color, -1)
            cv2.putText(display, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Threshold: {confidence_threshold:.2f}",
            f"Inference: {inference_time:.1f}ms"
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(display, line, (10, 25 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if detected_signs:
            signs_text = "Detected: " + ", ".join([f"{s[0]}" for s in detected_signs])
            cv2.putText(display, signs_text, (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Sign Language Detection (Swin-T)', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n👋 Thoát...")
            break
        elif key == ord('s'):
            filename = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, display)
            print(f"📸 Đã lưu: {filename}")
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"📈 Threshold: {confidence_threshold:.2f}")
        elif key == ord('-'):
            confidence_threshold = max(0.05, confidence_threshold - 0.05)
            print(f"📉 Threshold: {confidence_threshold:.2f}")
    cap.release()
    cv2.destroyAllWindows()
    print(" Done!")

if __name__ == "__main__":
    main()
