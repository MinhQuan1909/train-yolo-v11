"""
Thống kê VisDrone2019-DET dataset:
  - Số ảnh, số annotation file
  - Tổng số object và phân bố theo class

Usage:
  python utils/stats_dataset.py --src data/VisDrone2019-DET-train
"""

import argparse
import os
from collections import defaultdict

VISDRONE_CLASSES = {
    0: 'ignored',
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others',
}


def stats(src):
    img_dir = os.path.join(src, 'images')
    ann_dir = os.path.join(src, 'annotations')

    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]

    print(f'Ảnh       : {len(img_files)}')
    print(f'Annotation: {len(ann_files)}')

    class_counts = defaultdict(int)
    total_objects = 0
    empty_files = 0

    for fname in ann_files:
        path = os.path.join(ann_dir, fname)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.split(',')
            if len(parts) < 6:
                continue
            cls = int(parts[5])
            class_counts[cls] += 1
            total_objects += 1

    print(f'Tổng object : {total_objects}')
    print(f'Ảnh không có object: {empty_files}')
    print()
    print(f'{"Class":<6} {"Tên":<20} {"Số lượng":>10} {"Tỉ lệ":>8}')
    print('-' * 48)
    for cls_id in sorted(class_counts):
        name = VISDRONE_CLASSES.get(cls_id, f'class_{cls_id}')
        count = class_counts[cls_id]
        ratio = count / total_objects * 100
        print(f'{cls_id:<6} {name:<20} {count:>10,} {ratio:>7.2f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/VisDrone2019-DET-train')
    args = parser.parse_args()
    stats(args.src)


if __name__ == '__main__':
    main()
