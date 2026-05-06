"""
Convert VisDrone2019-DET dataset to COCO-style YOLO format.

VisDrone annotation format (comma-separated):
  bbox_left, bbox_top, bbox_width, bbox_height, score, class_id, truncation, occlusion

VisDrone class mapping → YOLO class index:
  0: ignored regions  → skip
  1: pedestrian       → 0
  2: people           → 1
  3: bicycle          → 2
  4: car              → 3
  5: van              → 4
  6: truck            → 5
  7: tricycle         → 6
  8: awning-tricycle  → 7
  9: bus              → 8
  10: motor           → 9
  11: others          → skip

Output structure:
  <dst>/
  ├── train2017.txt
  ├── val2017.txt
  ├── images/
  │   ├── train2017/   (symlinks to original images)
  │   └── val2017/
  └── labels/
      ├── train2017/   (converted YOLO .txt labels)
      └── val2017/

Usage:
  python utils/prepare_dataset.py \
      --src  data/VisDrone2019-DET-train \
      --dst  data/COCO \
      --val-ratio 0.1 \
      --seed 42
"""

import argparse
import os
import random
import shutil

from PIL import Image


# VisDrone class_id → YOLO class index (-1 = skip)
VISDRONE_TO_YOLO = {
    0: -1,   # ignored regions
    1:  0,   # pedestrian
    2:  1,   # people
    3:  2,   # bicycle
    4:  3,   # car
    5:  4,   # van
    6:  5,   # truck
    7:  6,   # tricycle
    8:  7,   # awning-tricycle
    9:  8,   # bus
    10: 9,   # motor
    11: -1,  # others
}


def convert_annotation(ann_path, img_w, img_h):
    """Convert one VisDrone annotation file to YOLO format lines."""
    lines = []
    with open(ann_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(',')
            if len(parts) < 6:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            visdrone_cls = int(parts[5])

            yolo_cls = VISDRONE_TO_YOLO.get(visdrone_cls, -1)
            if yolo_cls == -1:
                continue
            if w <= 0 or h <= 0:
                continue

            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines.append(f'{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
    return lines


def prepare(src, dst, val_ratio, seed):
    img_src = os.path.join(src, 'images')
    ann_src = os.path.join(src, 'annotations')

    # Collect all images that have a matching annotation file
    all_stems = []
    for fname in sorted(os.listdir(img_src)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue
        ann_file = os.path.join(ann_src, stem + '.txt')
        if os.path.exists(ann_file):
            all_stems.append(stem)

    print(f'Found {len(all_stems)} image-annotation pairs')

    random.seed(seed)
    random.shuffle(all_stems)
    n_val = int(len(all_stems) * val_ratio)
    val_stems = set(all_stems[:n_val])
    train_stems = all_stems[n_val:]

    print(f'Split: {len(train_stems)} train / {len(val_stems)} val')

    splits = {'train2017': train_stems, 'val2017': list(val_stems)}

    for split, stems in splits.items():
        img_dst_dir = os.path.join(dst, 'images', split)
        lbl_dst_dir = os.path.join(dst, 'labels', split)
        os.makedirs(img_dst_dir, exist_ok=True)
        os.makedirs(lbl_dst_dir, exist_ok=True)

        txt_lines = []
        for stem in stems:
            # Find original image file (handle any extension)
            img_file = None
            for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                candidate = os.path.join(img_src, stem + ext)
                if os.path.exists(candidate):
                    img_file = candidate
                    break
            if img_file is None:
                continue

            img_fname = os.path.basename(img_file)
            img_link = os.path.join(img_dst_dir, img_fname)

            # Symlink image (avoids copying large files)
            if not os.path.exists(img_link):
                os.symlink(os.path.abspath(img_file), img_link)

            # Get image dimensions for normalization
            with Image.open(img_file) as im:
                img_w, img_h = im.size

            # Convert annotation
            ann_path = os.path.join(ann_src, stem + '.txt')
            yolo_lines = convert_annotation(ann_path, img_w, img_h)

            lbl_path = os.path.join(lbl_dst_dir, stem + '.txt')
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            txt_lines.append(img_link)

        # Write index .txt
        index_path = os.path.join(dst, f'{split}.txt')
        with open(index_path, 'w') as f:
            f.write('\n'.join(txt_lines) + '\n')

        print(f'  {split}: wrote {len(txt_lines)} entries → {index_path}')

    print('\nDone. Update main.py:')
    print(f"  data_dir = '{os.path.abspath(dst)}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/VisDrone2019-DET-train',
                        help='VisDrone source directory (contains images/ and annotations/)')
    parser.add_argument('--dst', default='data/COCO',
                        help='Output directory (COCO-style structure)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Fraction of data to use for validation (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible split')
    args = parser.parse_args()

    prepare(args.src, args.dst, args.val_ratio, args.seed)


if __name__ == '__main__':
    main()
