import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import string
import cv2
import mediapipe as mp
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    from utils.logger import get_logger
except ImportError:
    # Fallback for running the file directly from src
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logger import get_logger


MODES = ("left", "right", "both")
WINDOW_CAPTURE = "Sign Dataset Capture"
WINDOW_REVIEW = "Sign Dataset Review"
WINDOW_REVIEW_GRID = "Sign Dataset Review Gallery"
FIXED_FRAME_WIDTH = 1280
FIXED_FRAME_HEIGHT = 720
DEFAULT_BLUR_THRESHOLD = 90.0
DEFAULT_HAND_APPEAR_DELAY = 3.0

# OpenCV waitKeyEx key codes vary by backend/OS; include common fallbacks.
KEY_LEFT = (2424832, 81)
KEY_RIGHT = (2555904, 83)
KEY_UP = (2490368, 82)
KEY_DOWN = (2621440, 84)
KEY_PAGEUP = (2162688,)
KEY_PAGEDOWN = (2228224,)


def normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace(" ", "_")


def list_dataset_classes(dataset_root: Path) -> List[str]:
    if not dataset_root.exists():
        return []

    return sorted(
        [
            p.name
            for p in dataset_root.iterdir()
            if p.is_dir() and p.name not in {"train", "test"}
        ]
    )


def prompt_class_name(dataset_root: Path) -> str:
    """Prompt user to create a new class or reuse an existing class."""
    dataset_root.mkdir(parents=True, exist_ok=True)

    while True:
        existing = list_dataset_classes(dataset_root)
        print("\n=== CHON CLASS ===")
        print("1) Tao class moi")
        print("2) Dung class cu")
        choice = input("Nhap lua chon (1/2): ").strip()

        if choice == "2":
            if not existing:
                print("Chua co class nao, vui long tao class moi.")
                continue

            print("\nDanh sach class hien co:")
            for idx, cls in enumerate(existing, start=1):
                print(f"{idx}) {cls}")

            selected = input("Chon so thu tu class: ").strip()
            if selected.isdigit() and 1 <= int(selected) <= len(existing):
                return existing[int(selected) - 1]

            print("Lua chon khong hop le, thu lai.")
            continue

        if choice == "1":
            raw_name = input("Nhap ten class moi: ").strip()
            class_name = normalize_class_name(raw_name)
            if class_name:
                return class_name

            print("Ten class khong duoc rong.")
            continue

        print("Lua chon khong hop le, vui long nhap 1 hoac 2.")


@dataclass
class HandDetection:
    handedness: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class ReviewState:
    image_path: Path
    label_path: Path
    mode: str


class ClassRegistry:
    """Keeps a stable class -> class_id mapping based on dataset folder names."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root

    def get_class_id(self, class_name: str) -> int:
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        class_dirs = sorted(
            [
                p.name
                for p in self.dataset_root.iterdir()
                if p.is_dir() and p.name not in {"train", "test"}
            ]
        )
        if class_name not in class_dirs:
            class_dirs.append(class_name)
            class_dirs = sorted(class_dirs)

        return class_dirs.index(class_name)


def ensure_class_structure(dataset_root: Path, class_name: str) -> Dict[str, Dict[str, Path]]:
    class_root = dataset_root / class_name
    paths: Dict[str, Dict[str, Path]] = {}

    for mode in MODES:
        images = class_root / mode / "images"
        labels = class_root / mode / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        paths[mode] = {"images": images, "labels": labels}

    return paths


class MediaPipeHandDetector:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        bbox_padding_px: int = 18,
    ) -> None:
        self._bbox_padding_px = bbox_padding_px
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def detect(self, frame_bgr) -> List[HandDetection]:
        image_h, image_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        detections: List[HandDetection] = []
        if not result.multi_hand_landmarks or not result.multi_handedness:
            return detections

        for landmarks, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
            score = float(handed.classification[0].score)
            hand_label = handed.classification[0].label

            x_vals = [lm.x for lm in landmarks.landmark]
            y_vals = [lm.y for lm in landmarks.landmark]

            x_min = max(0, int(min(x_vals) * image_w) - self._bbox_padding_px)
            y_min = max(0, int(min(y_vals) * image_h) - self._bbox_padding_px)
            x_max = min(image_w - 1, int(max(x_vals) * image_w) + self._bbox_padding_px)
            y_max = min(image_h - 1, int(max(y_vals) * image_h) + self._bbox_padding_px)

            if x_max <= x_min or y_max <= y_min:
                continue

            detections.append(
                HandDetection(
                    handedness=hand_label,
                    confidence=score,
                    bbox_xyxy=(x_min, y_min, x_max, y_max),
                )
            )

        return detections


def filter_by_mode(detections: List[HandDetection], mode: str) -> List[HandDetection]:
    if mode == "left":
        left = [d for d in detections if d.handedness.lower() == "left"]
        left.sort(key=lambda d: d.confidence, reverse=True)
        return left[:1]

    if mode == "right":
        right = [d for d in detections if d.handedness.lower() == "right"]
        right.sort(key=lambda d: d.confidence, reverse=True)
        return right[:1]

    if mode == "both":
        if len(detections) < 2:
            return []
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        return sorted_dets[:2]

    return []


def bbox_xyxy_to_yolo(
    bbox_xyxy: Tuple[int, int, int, int], image_w: int, image_h: int
) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox_xyxy

    width_px = max(1, x_max - x_min)
    height_px = max(1, y_max - y_min)
    center_x = x_min + width_px / 2.0
    center_y = y_min + height_px / 2.0

    x_center = max(0.0, min(1.0, center_x / image_w))
    y_center = max(0.0, min(1.0, center_y / image_h))
    width = max(0.0, min(1.0, width_px / image_w))
    height = max(0.0, min(1.0, height_px / image_h))

    return x_center, y_center, width, height


def yolo_to_bbox_xyxy(
    x_center: float, y_center: float, width: float, height: float, image_w: int, image_h: int
) -> Tuple[int, int, int, int]:
    w_px = max(1, int(width * image_w))
    h_px = max(1, int(height * image_h))
    c_x = int(x_center * image_w)
    c_y = int(y_center * image_h)

    x_min = max(0, c_x - w_px // 2)
    y_min = max(0, c_y - h_px // 2)
    x_max = min(image_w - 1, c_x + w_px // 2)
    y_max = min(image_h - 1, c_y + h_px // 2)

    return x_min, y_min, x_max, y_max


def save_yolo_labels(label_path: Path, class_id: int, yolo_boxes: List[Tuple[float, float, float, float]]) -> None:
    lines = [
        f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
        for x_c, y_c, w, h in yolo_boxes
    ]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def parse_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []

    parsed = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        x_c, y_c, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        parsed.append((cls_id, x_c, y_c, w, h))

    return parsed


class SignDatasetCollector:
    def __init__(
        self,
        dataset_root: Path,
        class_name: str,
        camera_id: int,
        num_per_mode: int,
        delay_seconds: int,
        capture_interval: float,
        blur_threshold: float,
        hand_appear_delay: float,
        dedup_hash_distance: int,
        dedup_window: int,
        enable_dedup: bool,
    ) -> None:
        self.dataset_root = dataset_root
        self.class_name = normalize_class_name(class_name)
        self.camera_id = camera_id
        self.num_per_mode = num_per_mode
        self.delay_seconds = delay_seconds
        self.capture_interval = capture_interval
        self.blur_threshold = blur_threshold
        self.hand_appear_delay = hand_appear_delay
        self.dedup_hash_distance = max(0, dedup_hash_distance)
        self.dedup_window = max(1, dedup_window)
        self.enable_dedup = enable_dedup
        self.recent_hashes_by_mode: Dict[str, List[int]] = {mode: [] for mode in MODES}

        self.logger = get_logger("dataset_collector")
        self.registry = ClassRegistry(dataset_root)
        self.class_paths = ensure_class_structure(dataset_root, self.class_name)
        self.class_id = self.registry.get_class_id(self.class_name)

        self.detector = MediaPipeHandDetector()
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera id {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FIXED_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FIXED_FRAME_HEIGHT)

    def close(self) -> None:
        self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()

    def _countdown(self, mode: str) -> None:
        if self.delay_seconds <= 0:
            return

        self.logger.capture(f"Preparing mode '{mode}' in {self.delay_seconds}s")
        for sec in range(self.delay_seconds, 0, -1):
            self.logger.capture(f"Mode {mode} starts in {sec}s...")
            time.sleep(1)

    def _is_blurry(self, frame) -> Tuple[bool, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return blur_score < self.blur_threshold, blur_score

    def _next_filename_stem(self, mode: str, index: int, frame=None) -> str:
        # timestamp + milliseconds
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)

        # random string
        rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

        # hash (nếu có frame)
        hash_part = ""
        if frame is not None:
            hash_part = hex(self._dhash(frame))[-6:]

        return f"{self.class_name}_{mode}_{ts}{ms}_{index:04d}_{rand_str}_{hash_part}"


    def _dhash(self, frame, hash_size: int = 8) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]

        fingerprint = 0
        for bit in diff.flatten():
            fingerprint = (fingerprint << 1) | int(bit)
        return fingerprint

    def _is_near_duplicate(self, frame, mode: str) -> Tuple[bool, Optional[int], int]:
        frame_hash = self._dhash(frame)
        recent = self.recent_hashes_by_mode.get(mode, [])
        if not recent:
            return False, None, frame_hash

        min_distance = min((frame_hash ^ old_hash).bit_count() for old_hash in recent)
        return min_distance <= self.dedup_hash_distance, min_distance, frame_hash

    def _remember_hash(self, mode: str, frame_hash: int) -> None:
        recent = self.recent_hashes_by_mode.setdefault(mode, [])
        recent.append(frame_hash)
        if len(recent) > self.dedup_window:
            del recent[0 : len(recent) - self.dedup_window]

    def run(self) -> None:
        self.logger.print_banner()
        self.logger.capture(
            f"Collecting class='{self.class_name}', class_id={self.class_id}, "
            f"num_per_mode={self.num_per_mode}, dedup={self.enable_dedup}, "
            f"dedup_hash_distance={self.dedup_hash_distance}, dedup_window={self.dedup_window}"
        )

        total_saved = 0
        total_blurry_skipped = 0
        total_duplicate_skipped = 0
        hand_visible_since: Optional[float] = None

        try:
            for mode in MODES:
                self._countdown(mode)
                saved = 0
                frame_index = 0
                last_capture_time = 0.0
                mode_duplicate_skipped = 0

                self.logger.capture(f"Start mode: {mode}")

                while saved < self.num_per_mode:
                    ok, frame = self.cap.read()
                    frame = cv2.flip(frame, 1)
                    if not ok:
                        self.logger.capture_error(self.class_name, "Failed to read frame from camera")
                        continue

                    frame = cv2.resize(frame, (FIXED_FRAME_WIDTH, FIXED_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

                    frame_index += 1
                    detections = self.detector.detect(frame)
                    selected = filter_by_mode(detections, mode)
                    eligible = bool(selected)
                    blurry, blur_score = self._is_blurry(frame)

                    now = time.time()
                    if eligible:
                        if hand_visible_since is None:
                            hand_visible_since = now
                        hand_visible_elapsed = now - hand_visible_since
                    else:
                        hand_visible_since = None
                        hand_visible_elapsed = 0.0

                    overlay = frame.copy()
                    for det in detections:
                        color = (0, 255, 255)
                        if det in selected:
                            color = (0, 255, 0)
                        x1, y1, x2, y2 = det.bbox_xyxy
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            overlay,
                            f"{det.handedness}:{det.confidence:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

                    cv2.putText(
                        overlay,
                        f"Class: {self.class_name} (id={self.class_id})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        overlay,
                        f"Mode: {mode} | Saved: {saved}/{self.num_per_mode}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        overlay,
                        "Press q to quit",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (180, 180, 180),
                        2,
                    )
                    cv2.putText(
                        overlay,
                        f"Blur score: {blur_score:.1f} (min {self.blur_threshold:.1f})",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (180, 180, 180),
                        2,
                    )
                    cv2.putText(
                        overlay,
                        f"Hand visible: {hand_visible_elapsed:.1f}/{self.hand_appear_delay:.1f}s",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (180, 180, 180),
                        2,
                    )
                    if blurry:
                        cv2.putText(
                            overlay,
                            "Blurry frame - auto skip",
                            (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2,
                        )
                    elif eligible and hand_visible_elapsed < self.hand_appear_delay:
                        cv2.putText(
                            overlay,
                            "Waiting for hand delay...",
                            (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 200, 0),
                            2,
                        )

                    if blurry:
                        total_blurry_skipped += 1

                    can_try_save = (
                        eligible
                        and not blurry
                        and hand_visible_since is not None
                        and hand_visible_elapsed >= self.hand_appear_delay
                        and (now - last_capture_time) >= self.capture_interval
                    )

                    near_duplicate = False
                    duplicate_distance: Optional[int] = None
                    current_hash: Optional[int] = None
                    if can_try_save and self.enable_dedup:
                        near_duplicate, duplicate_distance, computed_hash = self._is_near_duplicate(frame, mode)
                        current_hash = computed_hash
                        if near_duplicate:
                            total_duplicate_skipped += 1
                            mode_duplicate_skipped += 1

                    if near_duplicate and duplicate_distance is not None:
                        cv2.putText(
                            overlay,
                            f"Near-duplicate skipped (distance={duplicate_distance})",
                            (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 165, 255),
                            2,
                        )

                    if can_try_save and not near_duplicate:
                        image_h, image_w = frame.shape[:2]
                        yolo_boxes = [
                            bbox_xyxy_to_yolo(det.bbox_xyxy, image_w=image_w, image_h=image_h)
                            for det in selected
                        ]

                        stem = self._next_filename_stem(mode=mode, index=saved + 1,frame=frame)
                        image_path = self.class_paths[mode]["images"] / f"{stem}.jpg"
                        label_path = self.class_paths[mode]["labels"] / f"{stem}.txt"

                        cv2.imwrite(str(image_path), frame)
                        save_yolo_labels(label_path, self.class_id, yolo_boxes)

                        if self.enable_dedup:
                            if current_hash is None:
                                current_hash = self._dhash(frame)
                            self._remember_hash(mode, current_hash)

                        saved += 1
                        total_saved += 1
                        last_capture_time = time.time()
                        self.logger.capture_success(f"{self.class_name}/{mode}", saved)

                    cv2.imshow(WINDOW_CAPTURE, overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.logger.warning("Capture interrupted by user")
                        return

                self.logger.success(
                    f"Finished mode {mode}: {saved}/{self.num_per_mode} "
                    f"(duplicate skipped: {mode_duplicate_skipped})"
                )

        finally:
            self.close()
            self.logger.success(
                f"Capture done. Total saved images: {total_saved}. "
                f"Skipped blurry frames: {total_blurry_skipped}. "
                f"Skipped near-duplicate frames: {total_duplicate_skipped}"
            )


class SignDatasetReviewer:
    def __init__(
        self,
        dataset_root: Path,
        class_name: Optional[str],
        mode: Optional[str],
        start_index: int = 0,
        max_items: Optional[int] = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.class_name = normalize_class_name(class_name) if class_name else None
        self.mode = mode if mode in MODES else None
        self.start_index = max(0, start_index)
        self.max_items = max_items if (max_items is None or max_items > 0) else None

        self.logger = get_logger("dataset_reviewer")

        self._drawing = False
        self._draw_start: Optional[Tuple[int, int]] = None
        self._draw_end: Optional[Tuple[int, int]] = None
        self._edited_boxes: List[Tuple[int, int, int, int]] = []
        self._dirty = False
        self._grid_cell_rects: List[Tuple[int, int, int, int, int]] = []
        self._grid_selected_index: Optional[int] = None
        self._grid_open_requested = False
        self._ignore_gallery_keys_until_release = False

    def _collect_all_items(self) -> List[ReviewState]:
        classes = [self.class_name] if self.class_name else sorted(
            p.name for p in self.dataset_root.iterdir() if p.is_dir() and p.name not in {"train", "test"}
        )

        items: List[ReviewState] = []
        for cls in classes:
            class_root = self.dataset_root / cls
            if not class_root.exists():
                continue

            modes = [self.mode] if self.mode else list(MODES)
            for m in modes:
                img_dir = class_root / m / "images"
                lbl_dir = class_root / m / "labels"
                if not img_dir.exists():
                    continue

                for image_path in sorted(img_dir.glob("*.jpg")):
                    label_path = lbl_dir / f"{image_path.stem}.txt"
                    items.append(ReviewState(image_path=image_path, label_path=label_path, mode=m))

        return items

    def _collect_items(self) -> List[ReviewState]:
        items = self._collect_all_items()

        if self.max_items is None:
            return items[self.start_index :]

        end_index = self.start_index + self.max_items
        return items[self.start_index : end_index]

    def _load_boxes_from_label(
        self, label_path: Path, image_w: int, image_h: int
    ) -> List[Tuple[int, int, int, int]]:
        records = parse_yolo_labels(label_path)
        boxes: List[Tuple[int, int, int, int]] = []
        for _, x_c, y_c, w, h in records:
            boxes.append(yolo_to_bbox_xyxy(x_c, y_c, w, h, image_w, image_h))
        return boxes

    def _save_boxes_to_label(
        self,
        label_path: Path,
        class_id: int,
        boxes_xyxy: List[Tuple[int, int, int, int]],
        image_w: int,
        image_h: int,
    ) -> None:
        yolo_boxes = [bbox_xyxy_to_yolo(b, image_w=image_w, image_h=image_h) for b in boxes_xyxy]
        save_yolo_labels(label_path, class_id, yolo_boxes)

    def _mouse_callback(self, event, x, y, _flags, _params) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._draw_start = (x, y)
            self._draw_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._draw_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            self._draw_end = (x, y)

            if self._draw_start and self._draw_end:
                x1 = min(self._draw_start[0], self._draw_end[0])
                y1 = min(self._draw_start[1], self._draw_end[1])
                x2 = max(self._draw_start[0], self._draw_end[0])
                y2 = max(self._draw_start[1], self._draw_end[1])

                if x2 - x1 > 4 and y2 - y1 > 4:
                    self._edited_boxes.append((x1, y1, x2, y2))
                    self._dirty = True

    def _grid_mouse_callback(self, event, x, y, _flags, _params) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for idx, x1, y1, x2, y2 in self._grid_cell_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._grid_selected_index = idx
                self._grid_open_requested = True
                break

    def _fit_to_cell(self, image, cell_w: int, cell_h: int):
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        scale = min(cell_w / w, cell_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = 20 * np.ones((cell_h, cell_w, 3), dtype=np.uint8)
        x0 = (cell_w - new_w) // 2
        y0 = (cell_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _build_gallery(self, items: List[ReviewState], page: int):
        cols, rows = 4, 4
        per_page = cols * rows
        page_start = page * per_page
        page_items = items[page_start: page_start + per_page]

        pad = 10
        cell_w, cell_h = 300, 170
        header_h = 78
        footer_h = 30

        canvas_h = header_h + pad + rows * (cell_h + pad) + footer_h
        canvas_w = pad + cols * (cell_w + pad)
        canvas = 18 * np.ones((canvas_h, canvas_w, 3), dtype=np.uint8)

        self._grid_cell_rects = []

        total_pages = (len(items) + per_page - 1) // per_page if items else 1
        cv2.putText(
            canvas,
            f"Gallery page {page + 1}/{total_pages} | Total items: {len(items)}",
            (pad, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (230, 230, 230),
            2,
        )
        cv2.putText(
            canvas,
            "Click image to edit | enter=edit selected | arrows or n/p=next/prev page | q=quit",
            (pad, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (170, 170, 170),
            1,
        )

        for k, state in enumerate(page_items):
            r = k // cols
            c = k % cols
            x1 = pad + c * (cell_w + pad)
            y1 = header_h + pad + r * (cell_h + pad)
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            image = cv2.imread(str(state.image_path))
            box_count = 0
            if image is None:
                tile = 40 * np.ones((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(tile, "Missing image", (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                h, w = image.shape[:2]
                boxes = self._load_boxes_from_label(state.label_path, image_w=w, image_h=h)
                box_count = len(boxes)
                for x1b, y1b, x2b, y2b in boxes:
                    cv2.rectangle(image, (x1b, y1b), (x2b, y2b), (0, 220, 0), 2)
                tile = self._fit_to_cell(image, cell_w=cell_w, cell_h=cell_h)

            canvas[y1:y2, x1:x2] = tile

            abs_idx = page_start + k
            global_idx = abs_idx + 1
            label = f"#{global_idx} {state.mode}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 70), 2)
            cv2.putText(canvas, label, (x1 + 8, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            if image is not None:
                cv2.putText(canvas, f"boxes: {box_count}", (x1 + 8, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

            if self._grid_selected_index == abs_idx:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 3)

            self._grid_cell_rects.append((abs_idx, x1, y1, x2, y2))

        return canvas

    def _draw_overlay(
        self,
        image,
        boxes: List[Tuple[int, int, int, int]],
        mode: str,
        display_index: int,
        total: int,
    ):
        canvas = image.copy()

        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)

        if self._drawing and self._draw_start and self._draw_end:
            cv2.rectangle(canvas, self._draw_start, self._draw_end, (0, 140, 255), 2)

        cv2.putText(canvas, f"Mode: {mode}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Image: {display_index}/{total}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            canvas,
            "space=save back  n=next  p=prev  delete=delete  c=clear boxes  esc=back to gallery",
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (180, 180, 180),
            2,
        )
        cv2.putText(
            canvas,
            "Drag mouse to redraw/add bbox",
            (10, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (180, 180, 180),
            2,
        )

        return canvas

    def _review_one(self, items: List[ReviewState], index: int) -> Tuple[str, int]:
        state = items[index]
        image = cv2.imread(str(state.image_path))
        if image is None:
            self.logger.warning(f"Cannot open image: {state.image_path}")
            return "next", min(index + 1, max(0, len(items) - 1))

        h, w = image.shape[:2]
        records = parse_yolo_labels(state.label_path)
        class_id = records[0][0] if records else -1

        self._dirty = False
        self._edited_boxes = self._load_boxes_from_label(state.label_path, image_w=w, image_h=h)

        cv2.namedWindow(WINDOW_REVIEW)
        cv2.setMouseCallback(WINDOW_REVIEW, self._mouse_callback)

        while True:
            display_index = index + 1
            total = len(items)
            view = self._draw_overlay(
                image,
                self._edited_boxes,
                mode=state.mode,
                display_index=display_index,
                total=total,
            )
            cv2.imshow(WINDOW_REVIEW, view)
            key = cv2.waitKeyEx(20)

            is_delete = key in (3014656, 65535, 127, 8)

            if key == 27:  # ESC
                return "back_esc", index

            if key in (ord("c"), ord("C")):
                self._edited_boxes = []
                self._dirty = True
                continue

            if key in (ord("p"), ord("P"), *KEY_LEFT):  # P / left arrow
                return "prev", max(0, index - 1)

            if key in (ord("n"), ord("N"), *KEY_RIGHT):  # N / right arrow
                return "next", min(index + 1, max(0, len(items) - 1))

            if is_delete:
                try:
                    if state.image_path.exists():
                        state.image_path.unlink()
                    if state.label_path.exists():
                        state.label_path.unlink()
                    self.logger.warning(f"Deleted: {state.image_path.name}")
                    items.pop(index)
                    if not items:
                        return "empty", 0
                    return "back", min(index, len(items) - 1)
                except OSError as exc:
                    self.logger.error(f"Delete failed: {exc}")
                    continue

            if key in (13, 32):  # Enter or Space
                if class_id >= 0:
                    self._save_boxes_to_label(
                        label_path=state.label_path,
                        class_id=class_id,
                        boxes_xyxy=self._edited_boxes,
                        image_w=w,
                        image_h=h,
                    )
                    self.logger.success(f"Saved label: {state.label_path.name}")
                else:
                    self.logger.warning(
                        f"Skip save for {state.image_path.name}: no class_id in existing label"
                    )
                return "back", index

    def run(self) -> None:
        all_items = self._collect_all_items()
        total_items = len(all_items)
        if total_items == 0:
            self.logger.warning("No images found for review")
            return

        per_page = max(1, self.max_items or 16)
        if self.start_index >= total_items:
            self.logger.warning(
                f"start_index={self.start_index} is out of range (total={total_items}). "
                f"Set a lower --batch-page."
            )
            return

        items = all_items
        gallery_page = self.start_index // per_page

        end_index = min(total_items, (gallery_page + 1) * per_page)
        begin_index = gallery_page * per_page + 1
        self.logger.info(
            f"Reviewing {total_items} images total, showing page {gallery_page + 1} "
            f"(index {begin_index}-{end_index} / {total_items})"
        )

        cv2.namedWindow(WINDOW_REVIEW_GRID)
        cv2.setMouseCallback(WINDOW_REVIEW_GRID, self._grid_mouse_callback)

        if items:
            self._grid_selected_index = gallery_page * per_page

        while items:
            total_pages = (len(items) + per_page - 1) // per_page
            gallery_page = max(0, min(gallery_page, total_pages - 1))
            page_start = gallery_page * per_page

            if self._grid_selected_index is None:
                self._grid_selected_index = page_start
            self._grid_selected_index = max(0, min(self._grid_selected_index, len(items) - 1))

            gallery = self._build_gallery(items, gallery_page)
            cv2.imshow(WINDOW_REVIEW_GRID, gallery)

            key = cv2.waitKeyEx(30)

            if self._ignore_gallery_keys_until_release:
                if key == -1:
                    self._ignore_gallery_keys_until_release = False
                continue

            if key in (ord("q"), ord("Q")):
                self.logger.warning("Review interrupted by user")
                break

            if key == 27:
                # ESC in gallery is intentionally ignored to avoid accidental full exit.
                continue

            if key in (ord("n"), ord("N"), *KEY_RIGHT, *KEY_DOWN, *KEY_PAGEDOWN):
                if gallery_page < total_pages - 1:
                    gallery_page += 1
                    self._grid_selected_index = gallery_page * per_page
                continue

            if key in (ord("p"), ord("P"), *KEY_LEFT, *KEY_UP, *KEY_PAGEUP):
                if gallery_page > 0:
                    gallery_page -= 1
                    self._grid_selected_index = gallery_page * per_page
                continue

            if key in (13, 32) and self._grid_selected_index is not None:
                self._grid_open_requested = True

            if self._grid_open_requested and self._grid_selected_index is not None:
                self._grid_open_requested = False
                cursor = self._grid_selected_index

                while items:
                    action, cursor = self._review_one(items, cursor)

                    if action == "empty":
                        break
                    if action == "back_esc":
                        self._grid_selected_index = cursor
                        self._ignore_gallery_keys_until_release = True
                        break
                    if action == "back":
                        self._grid_selected_index = cursor
                        break
                    if action == "prev":
                        continue
                    if action == "next":
                        continue

                if not items:
                    break

                total_pages = (len(items) + per_page - 1) // per_page
                gallery_page = min(gallery_page, total_pages - 1)

        cv2.destroyWindow(WINDOW_REVIEW_GRID)
        try:
            cv2.destroyWindow(WINDOW_REVIEW)
        except cv2.error:
            pass
        self.logger.success("Review completed")


class SignDatasetCleaner:
    def __init__(
        self,
        dataset_root: Path,
        class_name: Optional[str],
        mode: Optional[str],
        blur_threshold: float,
        dedup_hash_distance: int,
        dedup_window: int,
        action: str,
        dry_run: bool,
    ) -> None:
        self.dataset_root = dataset_root
        self.class_name = normalize_class_name(class_name) if class_name else None
        self.mode = mode if mode in MODES else None
        self.blur_threshold = blur_threshold
        self.dedup_hash_distance = max(0, dedup_hash_distance)
        self.dedup_window = max(1, dedup_window)
        self.action = action
        self.dry_run = dry_run

        self.logger = get_logger("dataset_cleaner")
        self.filtered_root = self.dataset_root / "_filtered_out"

    def _dhash(self, frame, hash_size: int = 8) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]

        fingerprint = 0
        for bit in diff.flatten():
            fingerprint = (fingerprint << 1) | int(bit)
        return fingerprint

    def _is_blurry(self, frame) -> Tuple[bool, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return blur_score < self.blur_threshold, blur_score

    def _find_classes(self) -> List[str]:
        if self.class_name:
            return [self.class_name]

        if not self.dataset_root.exists():
            return []

        return sorted(
            p.name
            for p in self.dataset_root.iterdir()
            if p.is_dir() and p.name not in {"train", "test", "val", "_filtered_out"}
        )

    def _find_modes(self) -> List[str]:
        return [self.mode] if self.mode else list(MODES)

    def _move_or_delete(self, class_name: str, mode: str, reason: str, image_path: Path, label_path: Path) -> None:
        if self.dry_run:
            return

        if self.action == "delete":
            if image_path.exists():
                image_path.unlink()
            if label_path.exists():
                label_path.unlink()
            return

        target_base = self.filtered_root / reason / class_name / mode
        target_images = target_base / "images"
        target_labels = target_base / "labels"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        if image_path.exists():
            shutil.move(str(image_path), str(target_images / image_path.name))
        if label_path.exists():
            shutil.move(str(label_path), str(target_labels / label_path.name))

    def run(self) -> None:
        classes = self._find_classes()
        if not classes:
            self.logger.warning("No class folders found for cleaning")
            return

        total_images = 0
        kept_images = 0
        removed_blurry = 0
        removed_duplicate = 0
        missing_label = 0
        unreadable = 0

        for class_name in classes:
            class_root = self.dataset_root / class_name
            if not class_root.exists():
                continue

            for mode in self._find_modes():
                images_dir = class_root / mode / "images"
                labels_dir = class_root / mode / "labels"
                if not images_dir.exists() or not labels_dir.exists():
                    continue

                recent_hashes: List[int] = []
                mode_kept = 0
                mode_blurry = 0
                mode_duplicate = 0

                image_paths = sorted(p for p in images_dir.glob("*.jpg") if p.is_file())
                if not image_paths:
                    image_paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

                for image_path in image_paths:
                    total_images += 1
                    label_path = labels_dir / f"{image_path.stem}.txt"
                    if not label_path.exists():
                        missing_label += 1
                        continue

                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        unreadable += 1
                        continue

                    blurry, _ = self._is_blurry(frame)
                    if blurry:
                        removed_blurry += 1
                        mode_blurry += 1
                        self._move_or_delete(class_name, mode, "blurry", image_path, label_path)
                        continue

                    frame_hash = self._dhash(frame)
                    near_duplicate = False
                    if recent_hashes:
                        min_dist = min((frame_hash ^ old_hash).bit_count() for old_hash in recent_hashes)
                        near_duplicate = min_dist <= self.dedup_hash_distance

                    if near_duplicate:
                        removed_duplicate += 1
                        mode_duplicate += 1
                        self._move_or_delete(class_name, mode, "duplicate", image_path, label_path)
                        continue

                    kept_images += 1
                    mode_kept += 1
                    recent_hashes.append(frame_hash)
                    if len(recent_hashes) > self.dedup_window:
                        del recent_hashes[0 : len(recent_hashes) - self.dedup_window]

                self.logger.info(
                    f"Cleaned class={class_name}, mode={mode}: kept={mode_kept}, "
                    f"removed_blurry={mode_blurry}, removed_duplicate={mode_duplicate}"
                )

        self.logger.success(
            f"Clean summary: total={total_images}, kept={kept_images}, "
            f"removed_blurry={removed_blurry}, removed_duplicate={removed_duplicate}, "
            f"missing_label={missing_label}, unreadable={unreadable}, action={self.action}, dry_run={self.dry_run}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tool for collecting and reviewing sign language dataset with MediaPipe hand boxes"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Auto capture images + YOLO labels")
    capture.add_argument("--dataset-root", default="dataset", type=str)
    capture.add_argument("--class-name", type=str, default=None)
    capture.add_argument("--camera-id", type=int, default=0)
    capture.add_argument("--num-per-mode", type=int, default=30)
    capture.add_argument("--delay-seconds", type=int, default=0)
    capture.add_argument("--capture-interval", type=float, default=0.35)
    capture.add_argument("--blur-threshold", type=float, default=DEFAULT_BLUR_THRESHOLD)
    capture.add_argument("--hand-appear-delay", type=float, default=DEFAULT_HAND_APPEAR_DELAY)
    capture.add_argument(
        "--dedup-hash-distance",
        type=int,
        default=6,
        help="Maximum dHash Hamming distance considered near-duplicate (lower = stricter)",
    )
    capture.add_argument(
        "--dedup-window",
        type=int,
        default=8,
        help="Number of recent saved frames per mode used for duplicate checking",
    )
    capture.add_argument(
        "--disable-dedup",
        action="store_true",
        help="Disable near-duplicate frame filtering",
    )

    review = sub.add_parser("review", help="Review and edit boxes")
    review.add_argument("--dataset-root", default="dataset", type=str)
    review.add_argument("--class-name", type=str, default=None)
    review.add_argument("--mode", choices=list(MODES), default=None)
    review.add_argument("--batch-size", type=int, default=16)
    review.add_argument("--batch-page", type=int, default=1)
    review.add_argument("--all", action="store_true", help="Review all images (ignore batch options)")

    clean = sub.add_parser("clean", help="Post-process existing captures: remove blurry and near-duplicate images")
    clean.add_argument("--dataset-root", default="dataset", type=str)
    clean.add_argument("--class-name", type=str, default=None)
    clean.add_argument("--mode", choices=list(MODES), default=None)
    clean.add_argument("--blur-threshold", type=float, default=70.0)
    clean.add_argument("--dedup-hash-distance", type=int, default=6)
    clean.add_argument("--dedup-window", type=int, default=10)
    clean.add_argument("--action", choices=["move", "delete"], default="move")
    clean.add_argument("--dry-run", action="store_true", help="Preview what would be filtered without changing files")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    MY_CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    if args.command == "capture":
        dataset_root = Path(args.dataset_root)
        
        # Nếu người dùng truyền --class-name từ Terminal thì chụp 1 class đó
        if args.class_name:
            classes_to_run = [normalize_class_name(args.class_name)]
        else:
            print(f"Bắt đầu chụp tự động {len(MY_CLASSES)} classes...")
            classes_to_run = [normalize_class_name(c) for c in MY_CLASSES]

        for class_name in classes_to_run:
            print(f"\n--- ĐANG CHUẨN BỊ CLASS: {class_name} ---")
            
            # Khởi tạo và chạy collector cho từng class
            collector = SignDatasetCollector(
                dataset_root=dataset_root,
                class_name=class_name,
                camera_id=args.camera_id,
                num_per_mode=args.num_per_mode,
                delay_seconds=args.delay_seconds,
                capture_interval=args.capture_interval,
                blur_threshold=args.blur_threshold,
                hand_appear_delay=args.hand_appear_delay,
                dedup_hash_distance=args.dedup_hash_distance,
                dedup_window=args.dedup_window,
                enable_dedup=not args.disable_dedup,
            )
            collector.run()

        print("\nĐã hoàn thành toàn bộ danh sách!")

    if args.command == "capture":
        dataset_root = Path(args.dataset_root)
        fixed_class_name = normalize_class_name(args.class_name) if args.class_name else None

        while True:
            class_name = fixed_class_name or prompt_class_name(dataset_root)

            collector = SignDatasetCollector(
                dataset_root=dataset_root,
                class_name=class_name,
                camera_id=args.camera_id,
                num_per_mode=args.num_per_mode,
                delay_seconds=args.delay_seconds,
                capture_interval=args.capture_interval,
                blur_threshold=args.blur_threshold,
                hand_appear_delay=args.hand_appear_delay,
                dedup_hash_distance=args.dedup_hash_distance,
                dedup_window=args.dedup_window,
                enable_dedup=not args.disable_dedup,
            )
            collector.run()

            if fixed_class_name:
                break

            cont = input("Ban co muon tiep tuc voi class khac? (y/n): ").strip().lower()
            if cont not in {"y", "yes"}:
                break

    elif args.command == "review":
        max_items = None if args.all else max(1, int(args.batch_size))
        if max_items is None:
            start_index = 0
        else:
            page = max(1, int(args.batch_page))
            start_index = (page - 1) * max_items

        reviewer = SignDatasetReviewer(
            dataset_root=Path(args.dataset_root),
            class_name=args.class_name,
            mode=args.mode,
            start_index=start_index,
            max_items=max_items,
        )
        reviewer.run()

    elif args.command == "clean":
        cleaner = SignDatasetCleaner(
            dataset_root=Path(args.dataset_root),
            class_name=args.class_name,
            mode=args.mode,
            blur_threshold=args.blur_threshold,
            dedup_hash_distance=args.dedup_hash_distance,
            dedup_window=args.dedup_window,
            action=args.action,
            dry_run=args.dry_run,
        )
        cleaner.run()


if __name__ == "__main__":
    main()
