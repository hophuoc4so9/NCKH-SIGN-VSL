import cv2
import numpy as np
import os
from pathlib import Path

# ================= CẤU HÌNH =================
SPLIT = "test" 
DATASET_DIR = Path("dataset_resplit") / SPLIT
IMG_DIR = DATASET_DIR / "images"
LBL_DIR = DATASET_DIR / "labels"

CELL_SIZE = 200  # Khung hình sẽ là 800x800 pixel
COLS, ROWS = 4, 4
BATCH_SIZE = COLS * ROWS

valid_exts = ['.jpg', '.jpeg', '.png']
# ============================================

# Trạng thái toàn cục cho GUI
current_batch_images = []
current_batch_paths = []
selected_for_deletion = set()
grid_image_display = None

def read_yolo_label(txt_path, img_w, img_h):
    """Đọc file txt YOLO và chuyển sang tọa độ pixel."""
    boxes = []
    if not txt_path.exists(): return boxes
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cx, cy, w, h = map(float, parts[1:5])
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                boxes.append((x1, y1, x2, y2))
    return boxes

def draw_grid():
    """Vẽ 16 ảnh lên một khung hình lớn."""
    global grid_image_display
    
    grid_h = ROWS * CELL_SIZE
    grid_w = COLS * CELL_SIZE
    grid_image_display = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, (img, path) in enumerate(zip(current_batch_images, current_batch_paths)):
        col = idx % COLS
        row = idx // COLS
        x_offset = col * CELL_SIZE
        y_offset = row * CELL_SIZE
        
        display_img = img.copy()

        # Nếu ảnh bị chọn xóa, vẽ dấu X đỏ lớn chéo ảnh
        if idx in selected_for_deletion:
            cv2.line(display_img, (0, 0), (CELL_SIZE, CELL_SIZE), (0, 0, 255), 5)
            cv2.line(display_img, (CELL_SIZE, 0), (0, CELL_SIZE), (0, 0, 255), 5)
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (CELL_SIZE, CELL_SIZE), (0, 0, 255), -1)
            display_img = cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0)

        # Chèn ảnh nhỏ vào lưới lớn
        grid_image_display[y_offset:y_offset+CELL_SIZE, x_offset:x_offset+CELL_SIZE] = display_img

    cv2.imshow("Dataset Cleaner (800x800)", grid_image_display)

def mouse_callback(event, x, y, flags, param):
    """Xử lý sự kiện click chuột."""
    if event == cv2.EVENT_LBUTTONDOWN:
        col = x // CELL_SIZE
        row = y // CELL_SIZE
        idx = row * COLS + col
        
        if idx < len(current_batch_images):
            # Không cho phép chọn lại các ảnh đã bị xóa vật lý
            if np.array_equal(current_batch_images[idx][0, 0], [0, 0, 0]) and "DELETED" in str(current_batch_paths[idx]):
                pass 
            else:
                if idx in selected_for_deletion:
                    selected_for_deletion.remove(idx)
                else:
                    selected_for_deletion.add(idx)
                draw_grid()

def main():
    global current_batch_images, current_batch_paths, selected_for_deletion
    
    # CHỈ LẤY CÁC FILE CÓ TÊN BẮT ĐẦU BẰNG CHỮ "new" (không phân biệt hoa/thường)
    all_image_paths = [
        p for p in IMG_DIR.glob("*.*") 
        if p.suffix.lower() in valid_exts and p.name.lower().startswith('new')
    ]
    
    total_images = len(all_image_paths)
    print(f"Tổng số ảnh có chữ 'new' ở đầu trong tập {SPLIT}: {total_images}")

    if total_images == 0:
        print("Không tìm thấy ảnh nào bắt đầu bằng chữ 'new'. Thoát chương trình.")
        return

    cv2.namedWindow("Dataset Cleaner (800x800)")
    cv2.setMouseCallback("Dataset Cleaner (800x800)", mouse_callback)

    i = 0
    while i < total_images:
        current_batch_paths = all_image_paths[i:i+BATCH_SIZE]
        current_batch_images = []
        selected_for_deletion = set()

        for img_path in current_batch_paths:
            img = cv2.imread(str(img_path))
            
            # Xử lý trường hợp lùi lại trang cũ nhưng ảnh đã bị xóa vật lý
            if img is None:
                img_blank = np.zeros((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8)
                cv2.putText(img_blank, "DELETED", (CELL_SIZE//4, CELL_SIZE//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                current_batch_images.append(img_blank)
                continue

            h, w = img.shape[:2]
            txt_path = LBL_DIR / (img_path.stem + ".txt")
            boxes = read_yolo_label(txt_path, w, h)
            
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            img_resized = cv2.resize(img, (CELL_SIZE, CELL_SIZE))
            cv2.putText(img_resized, img_path.name, (5, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            current_batch_images.append(img_resized)

        draw_grid()

        print(f"\n--- Đang xem ảnh từ {i+1} đến {min(i+BATCH_SIZE, total_images)} ---")
        print("🖱️  Click chuột trái: Đánh dấu xóa")
        print("⌨️  [ENTER] hoặc [SPACE]: XÓA CÁC ẢNH ĐÃ CHỌN và qua trang")
        print("⌨️  [D]: Sang trang tiếp (Không xóa)")
        print("⌨️  [A]: Lùi lại trang trước (Không xóa)")
        print("⌨️  [Q]: Thoát")

        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # Bấm Enter (13), Space (32) để XÓA và đi tiếp
            if key in [13, 32]: 
                for idx in selected_for_deletion:
                    img_to_delete = current_batch_paths[idx]
                    txt_to_delete = LBL_DIR / (img_to_delete.stem + ".txt")
                    
                    try:
                        img_to_delete.unlink(missing_ok=True)
                        txt_to_delete.unlink(missing_ok=True)
                        print(f"🗑️ Đã xóa: {img_to_delete.name}")
                    except Exception as e:
                        pass
                
                i += BATCH_SIZE
                break
                
            # Bấm phím D (tiến)
            elif key in [ord('d'), ord('D')]:
                i += BATCH_SIZE
                break
                
            # Bấm phím A (lùi)
            elif key in [ord('a'), ord('A')]:
                i = max(0, i - BATCH_SIZE)
                break
                
            # Bấm Q hoặc ESC để thoát
            elif key in [113, 27, ord('Q')]:
                print("\nĐã thoát chương trình.")
                cv2.destroyAllWindows()
                return

    print("\n✅ Đã duyệt xong toàn bộ thư mục!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()