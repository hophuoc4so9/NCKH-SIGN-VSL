import os

# 1. Điền đường dẫn tuyệt đối TỚI TẬN FILE bị lỗi của bạn
FILE_PATH = r"D:\NCKH\SIGN-VSL\NCKH-SIGN-VSL\dataset_final\valid\labels\new_right_l_right_20260426_104618344_0008_j5u8w_cb8787.txt"

if not os.path.exists(FILE_PATH):
    print("❌ Không tìm thấy file! Bạn nhớ sửa lại đường dẫn FILE_PATH cho đúng thư mục chứa nó nhé.")
else:
    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            new_lines.append(line)
            continue
            
        class_id = parts[0]
        cx, cy, w, h = map(float, parts[1:])

        # Đổi ra 4 góc
        xmin = cx - w / 2.0
        ymin = cy - h / 2.0
        xmax = cx + w / 2.0
        ymax = cy + h / 2.0

        # Ép (Clamp) tọa độ để không bao giờ bị âm hoặc vượt quá 1.0
        xmin = max(0.0, min(1.0, xmin))
        ymin = max(0.0, min(1.0, ymin)) # Khắc phục triệt để lỗi -0.00000 của bạn
        xmax = max(0.0, min(1.0, xmax))
        ymax = max(0.0, min(1.0, ymax))

        # Tính ngược lại hệ YOLO
        new_w = xmax - xmin
        new_h = ymax - ymin
        new_cx = xmin + new_w / 2.0
        new_cy = ymin + new_h / 2.0

        new_line = f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n"
        new_lines.append(new_line)

    # Ghi đè lại file
    with open(FILE_PATH, 'w') as f:
        f.writelines(new_lines)

    print("✅ Đã fix xong! Tọa độ đã được gọt sát mép ảnh 0.0 cực kỳ hoàn hảo. Bạn có thể train DETR được rồi đó!")