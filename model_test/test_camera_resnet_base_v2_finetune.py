"""
Test Sign Language Detection bằng Camera
Sử dụng DETR Model trực tiếp

Cách sử dụng:
    python test_camera.py

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
from model_resnet import DETR

CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', '0', '1', '2', '3', '4']


MODEL_PATH = "D:\\NCKH\\SIGN-VSL\\NCKH-SIGN-VSL\\model_test\\best_signdetr_model_fine_tune.pth"
# E:\git\SignDETR\model_test\checkpoint_epoch_140.pth
# MODEL_PATH = "E:\\git\\SignDETR\\model_test\\checkpoint_epoch_140.pth"

CONFIDENCE_THRESHOLD = 0.5
NUM_CLASSES = 27


def get_colors(num_classes=22):
    """Tạo màu khác nhau cho mỗi class"""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors


def predict_full_frame(model, frame, transforms, device, threshold=0.3):
    """
    Dự đoán trực tiếp từ full frame (không cần MediaPipe)
    
    Returns:
        list of (class_name, confidence, bbox)
    """
    start_time = time.time()
    height, width = frame.shape[:2]
    
    with torch.no_grad():
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = transforms(image=rgb_image)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        result = model(input_tensor)
        
        probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]
        max_probs, max_classes = probabilities.max(-1)
        
        # Filter by threshold
        keep_mask = max_probs[0] > threshold
        
        detections = []
        for i in range(keep_mask.shape[0]):
            if keep_mask[i]:
                cls_idx = max_classes[0, i].item()
                conf = max_probs[0, i].item()
                
                # Get bbox
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
    print("   SIGN LANGUAGE DETECTION - Camera Test")
    print("=" * 60)
    
    # ========== SETUP DEVICE ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== LOAD SIGN LANGUAGE MODEL ==========
    print(f"\n📂 Loading model từ: {MODEL_PATH}")
    
    model = DETR(num_classes=NUM_CLASSES)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
            if 'loss' in checkpoint:
                print(f"   Loss: {checkpoint.get('loss'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        print("✅ Load model thành công!")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        print("   Hãy chắc chắn file model tồn tại và đúng format.")
        return
    
    model.to(device)
    model.eval()
    
    # ========== SETUP TRANSFORMS ==========
    transforms = A.Compose([
        A.Resize(512, 512), 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # ========== SETUP CAMERA ==========
    print("\n📷 Đang mở webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Camera: {frame_width}x{frame_height}")
    
    # Colors for visualization
    colors = get_colors(NUM_CLASSES)
    
    print("\n" + "=" * 60)
    print("   CONTROLS:")
    print("   q - Thoát")
    print("   s - Lưu screenshot")
    print("   +/- - Tăng/giảm confidence threshold")
    print("=" * 60 + "\n")
    
    # ========== MAIN LOOP ==========
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
        frame = cv2.flip(frame, 1)  # Mirror
        display = frame.copy()
        
        # ===== DETR Detection =====
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
            
            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display, (x1, y1-text_h-10), (x1+text_w+10, y1), color, -1)
            cv2.putText(display, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate FPS
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        # Draw info overlay
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Threshold: {confidence_threshold:.2f}",
            f"Inference: {inference_time:.1f}ms"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(display, line, (10, 25 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw detected signs
        if detected_signs:
            signs_text = "Detected: " + ", ".join([f"{s[0]}" for s in detected_signs])
            cv2.putText(display, signs_text, (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Sign Language Detection', display)
        
        # Handle key press
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
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!")


if __name__ == "__main__":
    main()
