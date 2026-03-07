import os
import shutil
import random

DATASET_DIR = r"E:\git\SignDETR\VIETNAM SIGN LANGUAGE.v7i.yolov8"
SPLITS = ["train", "valid", "test"]
LABELS_DIR_NAME = "labels"
IMAGES_DIR_NAME = "images"
OUTPUT_DIR = r"E:\git\SignDETR\sampled_labels"
SAMPLES_PER_LABEL = 10

# Danh sách nhãn
LABEL_NAMES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']

# Thống kê số lượng ảnh theo nhãn cho từng split
def count_images_per_label(split_dir):
    labels_path = os.path.join(split_dir, LABELS_DIR_NAME)
    images_path = os.path.join(split_dir, IMAGES_DIR_NAME)
    label_count = {name: 0 for name in LABEL_NAMES}
    label_to_images = {name: [] for name in LABEL_NAMES}

    for lbl_file in os.listdir(labels_path):
        lbl_path = os.path.join(labels_path, lbl_file)
        img_name_base = os.path.splitext(lbl_file)[0]
        # Tìm ảnh tương ứng
        img_candidates = [img_name_base + ext for ext in ['.jpg', '.png', '.jpeg']]
        img_file = None
        for candidate in img_candidates:
            candidate_path = os.path.join(images_path, candidate)
            if os.path.exists(candidate_path):
                img_file = candidate_path
                break
        if not img_file:
            continue
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                label_id = int(parts[0])
                if 0 <= label_id < len(LABEL_NAMES):
                    label_name = LABEL_NAMES[label_id]
                    label_count[label_name] += 1
                    label_to_images[label_name].append(img_file)
    return label_count, label_to_images

# Thống kê
for split in SPLITS:
    split_dir = os.path.join(DATASET_DIR, split)
    count, _ = count_images_per_label(split_dir)
    print(f"=== {split.upper()} ===")
    for label, num in count.items():
        print(f"  {label}: {num}")

# # Tạo thư mục mẫu
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# for label in LABEL_NAMES:
#     label_dir = os.path.join(OUTPUT_DIR, label)
#     os.makedirs(label_dir, exist_ok=True)

# # Lấy mẫu 10 ảnh mỗi nhãn từ tất cả splits
# all_label_to_images = {name: [] for name in LABEL_NAMES}
# for split in SPLITS:
#     split_dir = os.path.join(DATASET_DIR, split)
#     _, label_to_images = count_images_per_label(split_dir)
#     for label in LABEL_NAMES:
#         all_label_to_images[label].extend(label_to_images[label])

# for label in LABEL_NAMES:
#     images = all_label_to_images[label]
#     sample_imgs = random.sample(images, min(SAMPLES_PER_LABEL, len(images)))
#     for img_path in sample_imgs:
#         fname = os.path.basename(img_path)
#         dest = os.path.join(OUTPUT_DIR, label, fname)
#         shutil.copy(img_path, dest)
#     print(f"Đã copy {len(sample_imgs)} ảnh cho nhãn {label}.")
