import os
import shutil
import random

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ TỶ LỆ
# ==========================================
DIR_DATASET_NEW = r"D:\NCKH\SIGN-VSL\NCKH-SIGN-VSL\dataset"
DIR_DATASET_FILTERED = r"D:\NCKH\SIGN-VSL\NCKH-SIGN-VSL\dataset_filtered"
DIR_FINAL = r"D:\NCKH\SIGN-VSL\NCKH-SIGN-VSL\dataset_final"

# Tỷ lệ chia tập (Train 70%, Valid 20%, Test 10%)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
# Test sẽ lấy phần % còn lại

CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']
HAND_TYPES = ['both', 'left', 'right']

# Cố định seed để nếu bạn có chạy lại script thì nó vẫn chia bộ ảnh giống y hệt (không bị xáo trộn lung tung)
random.seed(42)

def create_yolo_structure(base_dir):
    """Tạo cấu trúc thư mục chuẩn YOLO cho dataset_final"""
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

def gather_data():
    """Gom toàn bộ đường dẫn ảnh và nhãn từ 2 dataset, phân loại theo Class"""
    all_data = {c: [] for c in CLASSES}
    
    # Danh sách các nguồn cần quét: (Đường dẫn, Tiền tố để đổi tên file)
    sources = [
        (DIR_DATASET_NEW, "new"),
        (DIR_DATASET_FILTERED, "filtered")
    ]
    
    for source_dir, source_prefix in sources:
        if not os.path.exists(source_dir):
            print(f"[!] Không tìm thấy thư mục: {source_dir}")
            continue
            
        for class_name in CLASSES:
            class_folder = class_name.lower()
            class_path = os.path.join(source_dir, class_folder)
            
            if not os.path.exists(class_path):
                continue
                
            for hand_type in HAND_TYPES:
                img_dir = os.path.join(class_path, hand_type, 'images')
                lbl_dir = os.path.join(class_path, hand_type, 'labels')
                
                if not os.path.exists(img_dir):
                    continue
                    
                for img_name in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_name)
                    
                    # Tìm file label tương ứng
                    lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
                    lbl_path = os.path.join(lbl_dir, lbl_name)
                    
                    if os.path.exists(lbl_path):
                        # Lưu trữ: (Đường dẫn ảnh, Đường dẫn nhãn, Tên mới dự kiến)
                        new_base_name = f"{source_prefix}_{hand_type}_{img_name}"
                        new_lbl_name = f"{source_prefix}_{hand_type}_{lbl_name}"
                        all_data[class_name].append((img_path, lbl_path, new_base_name, new_lbl_name))
                        
    return all_data

def main():
    print("=== BẮT ĐẦU GỘP VÀ CHIA DATASET ===")
    
    # 1. Tạo cấu trúc thư mục đích
    print("1. Đang tạo cấu trúc thư mục chuẩn YOLO...")
    create_yolo_structure(DIR_FINAL)
    
    # 2. Thu thập dữ liệu
    print("2. Đang quét và gom nhóm dữ liệu (left, right, both)...")
    data_dict = gather_data()
    
    total_train = total_val = total_test = 0
    
    # 3. Trộn và Copy
    print("3. Đang chia tập Train/Valid/Test và copy files...\n")
    for class_name in CLASSES:
        items = data_dict[class_name]
        total_items = len(items)
        
        if total_items == 0:
            print(f"[-] Class {class_name}: KHÔNG CÓ DỮ LIỆU!")
            continue
            
        # Trộn ngẫu nhiên
        random.shuffle(items)
        
        # Tính toán index để cắt mảng
        train_end = int(total_items * TRAIN_RATIO)
        valid_end = train_end + int(total_items * VALID_RATIO)
        
        # Chia mảng thành 3 tập
        train_items = items[:train_end]
        valid_items = items[train_end:valid_end]
        test_items = items[valid_end:]
        
        # Thống kê
        total_train += len(train_items)
        total_val += len(valid_items)
        total_test += len(test_items)
        
        # Hàm copy cục bộ
        def copy_items(item_list, split_folder):
            for img_src, lbl_src, img_new_name, lbl_new_name in item_list:
                img_dst = os.path.join(DIR_FINAL, split_folder, 'images', img_new_name)
                lbl_dst = os.path.join(DIR_FINAL, split_folder, 'labels', lbl_new_name)
                
                shutil.copy(img_src, img_dst)
                shutil.copy(lbl_src, lbl_dst)
                
        # Thực hiện copy
        copy_items(train_items, 'train')
        copy_items(valid_items, 'valid')
        copy_items(test_items, 'test')
        
        print(f"[+] Class {class_name}: Tổng {total_items} ảnh -> Train: {len(train_items)} | Valid: {len(valid_items)} | Test: {len(test_items)}")

    print("\n=== HOÀN TẤT! ĐÃ TẠO DATASET FINAL ===")
    print(f"Tổng quan toàn bộ Dataset Final:")
    print(f"- Tập TRAIN : {total_train} ảnh")
    print(f"- Tập VALID : {total_val} ảnh")
    print(f"- Tập TEST  : {total_test} ảnh")
    print(f"- Tổng cộng : {total_train + total_val + total_test} ảnh")
    print(f"-> Thư mục lưu trữ: {DIR_FINAL}")
    
    # Tạo thêm file data.yaml cho tiện train YOLO luôn
    yaml_path = os.path.join(DIR_FINAL, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.join(DIR_FINAL, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(DIR_FINAL, 'valid', 'images')}\n")
        f.write(f"test: {os.path.join(DIR_FINAL, 'test', 'images')}\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")
    print("\n[i] Đã tự động tạo file data.yaml trong dataset_final. Bạn có thể dùng luôn để train!")

if __name__ == "__main__":
    main()