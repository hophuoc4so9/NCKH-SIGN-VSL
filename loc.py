import os
import cv2
import numpy as np

# Thư mục chứa dataset bạn đang muốn lọc lại
NEW_DIR = r"D:\NCKH\SIGN-VSL\NCKH-SIGN-VSL\dataset_filtered"

# Cấu hình dữ liệu
CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']
HAND_TYPE = "both"

def main():
    print("=== CÔNG CỤ DUYỆT ẢNH BẰNG TAY ===")
    print("- Đang quét tìm các ảnh có nghi ngờ bị crop mảng đen/xám...")
    print("- HƯỚNG DẪN: Khi cửa sổ ảnh hiện lên:")
    print("  + Bấm phím 'd' trên bàn phím: ĐỂ XÓA ảnh (và label).")
    print("  + Bấm phím 'Space' (hoặc phím bất kỳ khác): ĐỂ GIỮ LẠI và xem ảnh tiếp theo.")
    print("  + Bấm phím 'q' (hoặc Esc): Để thoát chương trình sớm.")
    print("==================================\n")

    count_deleted = 0
    count_kept = 0

    for class_name in CLASSES:
        class_folder = class_name.lower()
        img_dir = os.path.join(NEW_DIR, class_folder, HAND_TYPE, 'images')
        lbl_dir = os.path.join(NEW_DIR, class_folder, HAND_TYPE, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            if not os.path.exists(lbl_path):
                continue
                
            # Đọc label
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                continue
                
            # Lấy thông số Bounding Box
            parts = lines[0].strip().split()
            if len(parts) < 5:
                continue
                
            _, x_center, y_center, bbox_w, bbox_h = map(float, parts[:5])
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Tính tọa độ pixel thực tế của Bounding Box
            x1 = int((x_center - bbox_w / 2) * w)
            y1 = int((y_center - bbox_h / 2) * h)
            x2 = int((x_center + bbox_w / 2) * w)
            y2 = int((y_center + bbox_h / 2) * h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
                
            roi_area = roi.shape[0] * roi.shape[1]
            
            # Áp dụng bộ lọc 15% (giống thuật toán trước) để tìm ảnh nghi ngờ
            _, mask_black = cv2.threshold(roi, 5, 255, cv2.THRESH_BINARY_INV)
            mask_gray = cv2.inRange(roi, 113, 115)
            mask_combined = cv2.bitwise_or(mask_black, mask_gray)
            
            contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            is_suspicious = False
            for cnt in contours:
                cx, cy, bounding_w, bounding_h = cv2.boundingRect(cnt)
                area = bounding_w * bounding_h
                if area > roi_area * 0.15:
                    is_suspicious = True
                    break
                    
            # NẾU PHÁT HIỆN ẢNH NGHI NGỜ -> HIỂN THỊ ĐỂ BẠN REVIEW
            if is_suspicious:
                display_img = img.copy()
                # Vẽ Bounding Box màu xanh lá
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Ghi chú lên ảnh (Góc trên bên trái)
                cv2.putText(display_img, f"Class: {class_name} | {img_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_img, "'d': XOA | Phim khac: GIU", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Hiển thị ảnh
                cv2.imshow("Review Dataset - NCKH", display_img)
                
                # Chờ bạn bấm phím
                key = cv2.waitKey(0) & 0xFF
                
                # Mã ASCII: 'd' là 100, 'q' là 113, Esc là 27
                if key == ord('q') or key == 27:
                    print("\n[!] Đã thoát chương trình sớm.")
                    cv2.destroyAllWindows()
                    return
                elif key == ord('d'):
                    # Xóa ảnh và file txt trực tiếp trên ổ cứng
                    os.remove(img_path)
                    os.remove(lbl_path)
                    count_deleted += 1
                    print(f"[-] Đã XÓA: {class_name}/{img_name}")
                else:
                    count_kept += 1
                    print(f"[+] Đã GIỮ: {class_name}/{img_name}")

    cv2.destroyAllWindows()
    print("\n=== HOÀN TẤT DUYỆT ẢNH ===")
    print(f"Tổng số ảnh bị xóa: {count_deleted}")
    print(f"Tổng số ảnh được giữ: {count_kept}")

if __name__ == "__main__":
    main()