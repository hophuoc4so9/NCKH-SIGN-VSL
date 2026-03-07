import cv2
import numpy as np
import os
import sys
import time
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Thêm thư mục src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_resnet import DETR

def get_colors(num_classes=22):
    """Tạo màu khác nhau cho mỗi class"""
    import colorsys
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors

def box_cxcywh_to_xyxy(x):
    """Chuyển đổi tọa độ từ (center_x, center_y, w, h) sang (x1, y1, x2, y2)"""
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox, size):
    """Scale tọa độ box từ [0, 1] về kích thước ảnh thực tế"""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox.device)
    return b

def main():
    # CONFIGURATION
    SIGN_MODEL_PATH = "pretrained/best_model_resnet.pt"
    NUM_CLASSES = 22
    CONFIDENCE_THRESHOLD = 0.5 
    
    CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']
   
    COLORS = get_colors(NUM_CLASSES)
    
    # 1. LOAD MODEL DETR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    sign_model = DETR(num_classes=NUM_CLASSES)
    
    try:
        checkpoint = torch.load(SIGN_MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            sign_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            sign_model.load_state_dict(checkpoint)
        print("✅ Load model DETR thành công!")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return
    
    sign_model.to(device)
    sign_model.eval()
    
    
    transforms = A.Compose([
        A.Resize(224, 224), 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # 3. MỞ WEBCAM
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("=" * 60)
    print("DETR REAL-TIME DETECTION (No MediaPipe)")
    print(f"Camera: {frame_width}x{frame_height}")
    print("Press 'q' to quit, '+/-' to adjust threshold")
    print("=" * 60)

    current_threshold = CONFIDENCE_THRESHOLD

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        
        frame = cv2.flip(frame, 1) # Mirror
        display = frame.copy()
        
        # Preprocess cho Model
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = transforms(image=rgb_image)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        start_time = time.time()
        
        # 4. INFERENCE
        with torch.no_grad():
            outputs = sign_model(input_tensor)
            
        inference_time = (time.time() - start_time) * 1000

        # 5. HẬU XỬ LÝ (Lấy logits và bboxes)
        # Lấy xác suất bằng softmax (loại bỏ class cuối cùng là 'no object' của DETR)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        
        # Chỉ giữ lại những dự đoán vượt ngưỡng threshold
        keep = probas.max(-1).values > current_threshold
        
        # Chuyển đổi tọa độ box về pixel thực tế
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], (frame_width, frame_height))
        probas_kept = probas[keep]

        # 6. VẼ BOX VÀ LABEL
        for p, (xmin, ymin, xmax, ymax) in zip(probas_kept, bboxes_scaled.tolist()):
            cl = p.argmax().item()
            conf = p[cl].item()
            color = COLORS[cl]
            
            # Vẽ Box
            cv2.rectangle(display, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            
            # Vẽ Nhãn
            label = f"{CLASSES[cl]}: {conf:.2f}"
            cv2.putText(display, label, (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Hiển thị thông tin hệ thống
        cv2.putText(display, f"FPS: {1000/inference_time:.1f} | Thr: {current_threshold:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('DETR Sign Language Detection', display)
        
        # Phím tắt điều khiển
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            current_threshold = min(0.95, current_threshold + 0.05)
        elif key == ord('-') or key == ord('_'):
            current_threshold = max(0.05, current_threshold - 0.05)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()