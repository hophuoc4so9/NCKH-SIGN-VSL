from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import albumentations as A


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODE_NAMES = {"left", "right", "both"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    class_name: str
    mode: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class ParsedLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    seed: int = 42

    def normalized(self) -> Tuple[float, float, float]:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive number")
        return self.train_ratio / total, self.val_ratio / total, self.test_ratio / total


@dataclass(frozen=True)
class AugmentConfig:
    copies_per_image: int = 2
    target_size: int = 512
    seed: int = 42
    min_visibility: float = 0.3
    save_originals: bool = False


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def find_image_for_stem(images_dir: Path, stem: str) -> Optional[Path]:
    for candidate in sorted(images_dir.glob(stem + ".*")):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def parse_label_file(label_path: Path) -> List[ParsedLabel]:
    if not label_path.exists():
        return []

    records: List[ParsedLabel] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue

        records.append(
            ParsedLabel(
                class_id=class_id,
                x_center=float(np.clip(x_center, 0.0, 1.0)),
                y_center=float(np.clip(y_center, 0.0, 1.0)),
                width=float(np.clip(width, 0.0, 1.0)),
                height=float(np.clip(height, 0.0, 1.0)),
            )
        )

    return records


def save_label_file(label_path: Path, records: Sequence[ParsedLabel]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{record.class_id} {record.x_center:.6f} {record.y_center:.6f} {record.width:.6f} {record.height:.6f}"
        for record in records
    ]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def collect_samples(source_root: Path) -> List[Sample]:
    samples: List[Sample] = []

    for class_dir in sorted(p for p in source_root.iterdir() if p.is_dir() and p.name not in SPLIT_NAMES):
        class_name = normalize_class_name(class_dir.name)

        nested_modes = [p for p in class_dir.iterdir() if p.is_dir() and p.name in MODE_NAMES]
        if nested_modes:
            for mode_dir in sorted(nested_modes):
                images_dir = mode_dir / "images"
                labels_dir = mode_dir / "labels"
                if not images_dir.exists() or not labels_dir.exists():
                    continue

                for image_path in sorted(p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)):
                    label_path = labels_dir / f"{image_path.stem}.txt"
                    if label_path.exists():
                        samples.append(
                            Sample(
                                class_name=class_name,
                                mode=mode_dir.name,
                                image_path=image_path,
                                label_path=label_path,
                            )
                        )
        else:
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                continue

            for image_path in sorted(p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)):
                label_path = labels_dir / f"{image_path.stem}.txt"
                if label_path.exists():
                    samples.append(
                        Sample(
                            class_name=class_name,
                            mode="flat",
                            image_path=image_path,
                            label_path=label_path,
                        )
                    )

    return samples


def split_indices(total: int, config: SplitConfig) -> Dict[str, List[int]]:
    train_ratio, val_ratio, test_ratio = config.normalized()
    indices = list(range(total))
    rng = random.Random(config.seed)
    rng.shuffle(indices)

    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    if train_count + val_count > total:
        overflow = train_count + val_count - total
        if val_count >= overflow:
            val_count -= overflow
        else:
            train_count = max(0, train_count - (overflow - val_count))
            val_count = 0
    test_count = total - train_count - val_count

    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:train_count + val_count + test_count]

    return {"train": train_indices, "val": val_indices, "test": test_indices}


def destination_stem(sample: Sample, source_index: int) -> str:
    mode_part = sample.mode if sample.mode != "flat" else "all"
    return f"{sample.class_name}_{mode_part}_{sample.image_path.stem}_{source_index:04d}"


def copy_split_samples(samples: Sequence[Sample], output_root: Path, config: SplitConfig) -> Dict[str, int]:
    split_map = split_indices(len(samples), config)
    counts = {name: 0 for name in SPLIT_NAMES}

    for split_name, indices in split_map.items():
        images_out = output_root / split_name / "images"
        labels_out = output_root / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for order, sample_index in enumerate(indices):
            sample = samples[sample_index]
            new_stem = destination_stem(sample, sample_index)

            image_dest = images_out / f"{new_stem}{sample.image_path.suffix.lower()}"
            label_dest = labels_out / f"{new_stem}.txt"

            shutil.copy2(sample.image_path, image_dest)
            shutil.copy2(sample.label_path, label_dest)
            counts[split_name] += 1

    return counts


def make_train_augmenter(target_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=target_size),
            A.PadIfNeeded(min_height=target_size, min_width=target_size, border_mode=cv2.BORDER_CONSTANT, fill=(0, 0, 0)),
            A.RandomSizedBBoxSafeCrop(width=target_size, height=target_size, erosion_rate=0.0, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.Resize(target_size, target_size),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def augment_train_split(train_root: Path, config: AugmentConfig) -> Dict[str, int]:
    images_dir = train_root / "images"
    labels_dir = train_root / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing train/images or train/labels under {train_root}")

    rng = random.Random(config.seed)
    augmenter = make_train_augmenter(config.target_size)
    created = 0
    skipped = 0

    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        image_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{label_path.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            skipped += 1
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            skipped += 1
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        records = parse_label_file(label_path)
        if not records:
            skipped += 1
            continue

        bboxes = [[r.x_center, r.y_center, r.width, r.height] for r in records]
        class_labels = [r.class_id for r in records]

        if config.save_originals:
            output_base = label_path.stem
            copy_image_dest = images_dir / f"{output_base}_orig{image_path.suffix.lower()}"
            copy_label_dest = labels_dir / f"{output_base}_orig.txt"
            if not copy_image_dest.exists():
                shutil.copy2(image_path, copy_image_dest)
                shutil.copy2(label_path, copy_label_dest)

        for aug_idx in range(config.copies_per_image):
            try:
                transformed = augmenter(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
            except Exception:
                skipped += 1
                continue

            if len(transformed["bboxes"]) == 0:
                skipped += 1
                continue

            aug_stem = f"{label_path.stem}_aug{aug_idx + 1}"
            aug_image_bgr = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
            aug_image_path = images_dir / f"{aug_stem}{image_path.suffix.lower()}"
            aug_label_path = labels_dir / f"{aug_stem}.txt"

            cv2.imwrite(str(aug_image_path), aug_image_bgr)
            aug_records = [
                ParsedLabel(
                    class_id=int(cls_id),
                    x_center=float(box[0]),
                    y_center=float(box[1]),
                    width=float(box[2]),
                    height=float(box[3]),
                )
                for box, cls_id in zip(transformed["bboxes"], transformed["class_labels"])
            ]
            save_label_file(aug_label_path, aug_records)
            created += 1

    return {"created": created, "skipped": skipped}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DETR dataset: split and augment sign language samples")
    sub = parser.add_subparsers(dest="command", required=True)

    split_parser = sub.add_parser("split", help="Merge class folders and split into train/val/test")
    split_parser.add_argument("--source-root", type=str, default="dataset")
    split_parser.add_argument("--output-root", type=str, default="data")
    split_parser.add_argument("--train-ratio", type=float, default=0.7)
    split_parser.add_argument("--val-ratio", type=float, default=0.2)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--clean", action="store_true", help="Remove output root before writing")

    aug_parser = sub.add_parser("augment", help="Create augmented copies inside train split only")
    aug_parser.add_argument("--train-root", type=str, default="data/train")
    aug_parser.add_argument("--copies-per-image", type=int, default=2)
    aug_parser.add_argument("--target-size", type=int, default=512)
    aug_parser.add_argument("--seed", type=int, default=42)
    aug_parser.add_argument("--save-originals", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "split":
        source_root = Path(args.source_root)
        output_root = Path(args.output_root)
        if args.clean and output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        samples = collect_samples(source_root)
        if not samples:
            raise SystemExit(f"No samples found under {source_root}")

        config = SplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        counts = copy_split_samples(samples, output_root, config)

        summary = {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "total_samples": len(samples),
            "counts": counts,
            "ratios": config.normalized(),
        }
        (output_root / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    if args.command == "augment":
        train_root = Path(args.train_root)
        config = AugmentConfig(
            copies_per_image=max(0, args.copies_per_image),
            target_size=max(1, args.target_size),
            seed=args.seed,
            save_originals=bool(args.save_originals),
        )
        result = augment_train_split(train_root, config)
        print(json.dumps({"train_root": str(train_root), **result}, indent=2))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
