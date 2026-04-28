import os
import shutil
import cv2
import numpy as np

# Cấu hình đường dẫn
OLD_DIR = r"D:\NCKH\SIGN-VSL\NCKH-SIGN-VSL\dataset_old"
NEW_DIR = r"D:\NCKH\SIGN-VSL\NCKH-SIGN-VSL\dataset_filtered"

# Cấu hình dữ liệu
CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']
HAND_TYPE = "both"

# Thiết lập số lượng mục tiêu cho từng class (M = 60, còn lại = 50)
TARGET_COUNTS = {c: 50 for c in CLASSES}
TARGET_COUNTS['M'] = 60

# Dictionary theo dõi số lượng ảnh đã lấy
count_per_class = {c: 0 for c in CLASSES}

def is_augmented(img_path, yolo_line):
    """
    1. Kiểm tra viền đen xoay ảnh (toàn bộ ảnh).
    2. Kiểm tra vết crop/xóa (CHỈ TRONG Bounding Box).
    """
    img = cv2.imread(img_path)
    if img is None:
        return True 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Kiểm tra viền đen ở 4 góc (do xoay ảnh sinh ra) - Vẫn áp dụng cho toàn ảnh
    corners = [gray[0, 0], gray[0, w-1], gray[h-1, 0], gray[h-1, w-1]]
    if any(pixel_value < 5 for pixel_value in corners):
        return True
        
    # 2. Xử lý Bounding Box
    parts = yolo_line.strip().split()
    if len(parts) >= 5:
        # Bỏ qua index (parts[0]), lấy 4 thông số tọa độ
        _, x_center, y_center, bbox_w, bbox_h = map(float, parts[:5])
        
        # Chuyển đổi tọa độ chuẩn hóa YOLO (0-1) về pixel thực tế của OpenCV
        x1 = int((x_center - bbox_w / 2) * w)
        y1 = int((y_center - bbox_h / 2) * h)
        x2 = int((x_center + bbox_w / 2) * w)
        y2 = int((y_center + bbox_h / 2) * h)
        
        # Đảm bảo tọa độ không vượt quá viền ảnh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Cắt lấy vùng ảnh chỉ chứa vật thể (ROI - Region of Interest)
        roi = gray[y1:y2, x1:x2]
        
        # Bỏ qua nếu lỗi tọa độ khiến ROI rỗng
        if roi.size == 0:
            return False 
            
        roi_area = roi.shape[0] * roi.shape[1]
        
        # Áp dụng bộ lọc Cutout/Random Erasing CHỈ TRÊN VÙNG ROI
        _, mask_black = cv2.threshold(roi, 5, 255, cv2.THRESH_BINARY_INV)
        mask_gray = cv2.inRange(roi, 113, 115)
        
        mask_combined = cv2.bitwise_or(mask_black, mask_gray)
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            cx, cy, bounding_w, bounding_h = cv2.boundingRect(cnt)
            area = bounding_w * bounding_h
            
            # Kiểm tra: Mảng đen/xám chiếm > 30% diện tích Bounding Box
            if area > roi_area * 0.3:
                return True # Phát hiện crop bên trong bounding box -> Lọc bỏ
                
    return False # Ảnh an toàn

def process_split(split_name, check_augmentation=False):
    """Xử lý từng thư mục (test, valid, train)"""
    img_dir_old = os.path.join(OLD_DIR, split_name, 'images')
    lbl_dir_old = os.path.join(OLD_DIR, split_name, 'labels')
    
    if not os.path.exists(img_dir_old):
        return
        
    print(f"\n--- Đang quét thư mục: {split_name.upper()} (Có bật bộ lọc: {'CÓ' if check_augmentation else 'KHÔNG'}) ---")
    
    for img_name in os.listdir(img_dir_old):
        if all(count_per_class[c] >= TARGET_COUNTS[c] for c in CLASSES):
            return

        img_path = os.path.join(img_dir_old, img_name)
        lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
        lbl_path = os.path.join(lbl_dir_old, lbl_name)
        
        if not os.path.exists(lbl_path):
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        try:
            class_idx = int(lines[0].split()[0])
            class_name = CLASSES[class_idx]
        except (IndexError, ValueError):
            continue
            
        if count_per_class[class_name] >= TARGET_COUNTS[class_name]:
            continue
            
        # Nâng cấp: Truyền thêm dòng đầu tiên của file txt (chứa bounding box) vào hàm
        if check_augmentation and is_augmented(img_path, lines[0]):
            continue
            
        class_folder = class_name.lower()
        target_img_dir = os.path.join(NEW_DIR, class_folder, HAND_TYPE, 'images')
        target_lbl_dir = os.path.join(NEW_DIR, class_folder, HAND_TYPE, 'labels')
        
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)
        
        shutil.copy(img_path, os.path.join(target_img_dir, img_name))
        shutil.copy(lbl_path, os.path.join(target_lbl_dir, lbl_name))
        
        count_per_class[class_name] += 1
        print(f"Đã copy {class_name}: {count_per_class[class_name]}/{TARGET_COUNTS[class_name]} ({img_name})")

def main():
    process_split('valid', check_augmentation=False)
    process_split('test', check_augmentation=False)
    
    if not all(count_per_class[c] >= TARGET_COUNTS[c] for c in CLASSES):
        process_split('train', check_augmentation=True)

    print("\n--- BÁO CÁO THỐNG KÊ ---")
    for c in CLASSES:
        status = "ĐỦ" if count_per_class[c] >= TARGET_COUNTS[c] else "THIẾU"
        print(f"Class {c}: {count_per_class[c]}/{TARGET_COUNTS[c]} ảnh ({status})")
        
    if all(count_per_class[c] >= TARGET_COUNTS[c] for c in CLASSES):
        print("\n=> THÀNH CÔNG: Đã lấy đủ dữ liệu sạch cho tất cả các class theo yêu cầu!")
    else:
        print("\n=> LƯU Ý: Một số class không đủ ảnh dù đã quét hết dataset_old.")

if __name__ == "__main__":
    main()