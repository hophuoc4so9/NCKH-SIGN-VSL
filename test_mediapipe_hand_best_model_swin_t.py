"""
Sign Language  
Detect bàn tay bằng MediaPipe


Cách chạy:
    python test_mediapipe_hand.py

Nhấn 'q' để thoát.
"""

import cv2
import numpy as np
import urllib.request
import os
import sys
import time

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# PyTorch
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Thêm thư mục src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_swin_t import DETR


def download_model(url, filename):
    """Download model nếu chưa có"""
    if not os.path.exists(filename):
        print(f"📥 Đang download {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("✅ Download xong!")
    return filename


def get_hand_bbox(hand_landmarks, width, height, padding=30):
    """Lấy bounding box từ hand landmarks"""
    x_coords = [lm.x * width for lm in hand_landmarks]
    y_coords = [lm.y * height for lm in hand_landmarks]
    
    x1 = max(0, int(min(x_coords) - padding))
    y1 = max(0, int(min(y_coords) - padding))
    x2 = min(width, int(max(x_coords) + padding))
    y2 = min(height, int(max(y_coords) + padding))
    
    return x1, y1, x2, y2


def draw_hand_landmarks(image, hand_landmarks, color=(0, 255, 0)):
    """Vẽ landmarks và connections của bàn tay"""
    height, width = image.shape[:2]
    
    # Danh sách connections giữa các landmarks
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Ngón cái
        (0, 5), (5, 6), (6, 7), (7, 8),      # Ngón trỏ
        (0, 9), (9, 10), (10, 11), (11, 12), # Ngón giữa
        (0, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
        (0, 17), (17, 18), (18, 19), (19, 20), # Ngón út
        (5, 9), (9, 13), (13, 17)            # Lòng bàn tay
    ]
    
    # Lấy tọa độ pixel
    points = []
    for lm in hand_landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        points.append((px, py))
    
    # Vẽ connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(image, points[start_idx], points[end_idx], color, 2)
    
    # Vẽ landmarks
    for i, (px, py) in enumerate(points):
        # Màu khác cho đầu ngón tay
        if i in [4, 8, 12, 16, 20]:
            cv2.circle(image, (px, py), 6, (0, 0, 255), -1)  # Đỏ
        else:
            cv2.circle(image, (px, py), 4, color, -1)
    
    return image


def get_colors(num_classes=22):
    """Tạo màu khác nhau cho mỗi class"""
    import colorsys
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors


def predict_sign(model, hand_image, transforms, device, classes, threshold=0.3):
    """
    Dự đoán ký hiệu từ hình ảnh bàn tay
    Returns: (class_name, confidence, inference_time_ms) hoặc (None, None, time)
    """
    start_time = time.time()
    
    with torch.no_grad():
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
        
        # Transform
        transformed = transforms(image=rgb_image)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Inference
        result = model(input_tensor)
        
        # Xử lý kết quả
        probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]
        max_probs, max_classes = probabilities.max(-1)
        
        # Lấy prediction tốt nhất
        best_prob, best_idx = max_probs[0].max(dim=0)
        best_class = max_classes[0, best_idx]
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    if best_prob.item() > threshold:
        return classes[best_class.item()], best_prob.item(), inference_time
    
    return None, None, inference_time


def main():
    # ========== CẤU HÌNH ========== pretrained\best_model_swin_t.pt
    SIGN_MODEL_PATH = "pretrained/best_model_swin_t.pt"
    NUM_CLASSES = 22
    CONFIDENCE_THRESHOLD = 0.3
    HAND_PADDING = 30
    
    CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']
    COLORS = get_colors(NUM_CLASSES)
    
    print("=" * 60)
    print("🤟 SIGN LANGUAGE DETECTION - MediaPipe + DETR")
    print("=" * 60)
    
    # Download model
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"
    download_model(MODEL_URL, MODEL_PATH)
    
    # Tạo HandLandmarker
    print("🔧 Đang khởi tạo HandLandmarker...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("✅ MediaPipe sẵn sàng!")
    
    # ========== LOAD SIGN LANGUAGE MODEL ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    print(f"📦 Đang load model từ: {SIGN_MODEL_PATH}")
    sign_model = DETR(num_classes=NUM_CLASSES)
    
    try:
        checkpoint = torch.load(SIGN_MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            sign_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   📊 Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"   📉 Loss: {checkpoint.get('loss', 'N/A'):.4f}")
        else:
            sign_model.load_state_dict(checkpoint)
        print("✅ Load model thành công!")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return
    
    sign_model.to(device)
    sign_model.eval()
    
    # Transforms cho model
    transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Mở webcam
    print("📷 Đang mở webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐 Camera: {frame_width}x{frame_height}")
    print("=" * 60)
    print("🎯 'q' - Thoát")
    print("🎯 's' - Lưu ảnh bàn tay")
    print("🎯 '+'/'-' - Tăng/giảm threshold")
    print("=" * 60)
    
    frame_count = 0
    confidence_threshold = CONFIDENCE_THRESHOLD
    avg_inference_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)  # Mirror
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tạo MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        result = detector.detect(mp_image)
        
        # Copy frame để vẽ
        display = frame.copy()
        
        # Vẽ kết quả
        num_hands = len(result.hand_landmarks)
        detected_signs = []
        total_inference_time = 0
        
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            # Lấy handedness (Left/Right)
            handedness = result.handedness[idx][0]
            hand_label = handedness.category_name
            hand_score = handedness.score
            
            # Lấy bounding box
            x1, y1, x2, y2 = get_hand_bbox(hand_landmarks, frame_width, frame_height, padding=HAND_PADDING)
            
            # Crop ảnh bàn tay
            hand_image = frame[y1:y2, x1:x2]
            
            # Dự đoán sign language
            sign_class = None
            sign_conf = None
            inference_time = 0
            
            if hand_image.shape[0] > 20 and hand_image.shape[1] > 20:
                sign_class, sign_conf, inference_time = predict_sign(
                    sign_model, hand_image, transforms, device, 
                    CLASSES, threshold=confidence_threshold
                )
                total_inference_time += inference_time
            
            # Màu theo kết quả dự đoán
            if sign_class:
                class_idx = CLASSES.index(sign_class)
                color = COLORS[class_idx]
                detected_signs.append(f"{sign_class}({sign_conf:.2f})")
            else:
                color = (128, 128, 128)  # Xám nếu không detect được
            
            # Vẽ landmarks
            draw_hand_landmarks(display, hand_landmarks, color)
            
            # Vẽ bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Label - hiển thị sign hoặc hand type
            if sign_class:
                label = f"{sign_class}: {sign_conf:.2f}"
            else:
                label = f"{hand_label} (?)"
            cv2.putText(display, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Cập nhật average inference time
        if num_hands > 0:
            avg_inference_time = total_inference_time / num_hands
        
        # Log ra console
        if detected_signs:
            print(f"🤟 Frame {frame_count}: {', '.join(detected_signs)} | ⏱️ {avg_inference_time:.1f}ms")
        
        # Info - hiển thị nhiều thông tin hơn
        info_line1 = f"Hands: {num_hands} | Threshold: {confidence_threshold:.2f}"
        info_line2 = f"Inference: {avg_inference_time:.1f}ms"
        cv2.putText(display, info_line1, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, info_line2, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Hiển thị detected signs lớn ở dưới màn hình
        if detected_signs:
            big_text = " | ".join([s.split('(')[0] for s in detected_signs])
            (text_width, text_height), _ = cv2.getTextSize(
                big_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4
            )
            text_x = (frame_width - text_width) // 2
            text_y = frame_height - 50
            
            # Background
            cv2.rectangle(
                display,
                (text_x - 20, text_y - text_height - 20),
                (text_x + text_width + 20, text_y + 20),
                (0, 0, 0),
                -1
            )
            
            # Text
            cv2.putText(
                display, 
                big_text, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2.5, 
                (0, 255, 255), 
                4
            )
        
        cv2.putText(display, "Press 'q' quit | 's' save | '+'/'-' threshold", 
                    (10, frame_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('MediaPipe Hand Detection', display)
        
        # Phím tắt
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Thoát...")
            break
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"📈 Threshold: {confidence_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            confidence_threshold = max(0.05, confidence_threshold - 0.05)
            print(f"📉 Threshold: {confidence_threshold:.2f}")
        elif key == ord('s'):
            # Lưu ảnh bàn tay
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                x1, y1, x2, y2 = get_hand_bbox(hand_landmarks, frame_width, frame_height)
                hand_img = frame[y1:y2, x1:x2]
                if hand_img.size > 0:
                    path = f"hand_{frame_count}_{idx}.jpg"
                    cv2.imwrite(path, hand_img)
                    print(f"💾 Saved: {path}")
    
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!")


if __name__ == "__main__":
    main()
