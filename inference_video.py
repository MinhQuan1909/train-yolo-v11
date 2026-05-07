import os
import cv2
import torch
import numpy
import tqdm
import yaml
import time
from nets import nn
from utils import util
from utils.dataset import resize

def main():
    video_path = 'data_val/drone-aerials-of-brazil-flooding-devastation.mp4'
    output_dir = 'data_val/result_images'
    os.makedirs(output_dir, exist_ok=True)

    conf_thres = 0.1
    iou_thres = 0.2
    fps = 30
    output_path = f'output_video-confidence_threshold_{conf_thres}-iou_threshold_{iou_thres}.mp4'
    batch_size = 64

    # Load model
    print("Loading model...")
    model = torch.load('./weights/best.pt', map_location='cuda', weights_only=False)
    model = model['model'].float().fuse().half().eval()

    # Load class names
    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
    names = params['names']

    # Đọc frames từ video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không mở được video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames, FPS gốc: {fps:.1f}")

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((f"frame_{frame_idx:06d}.jpg", frame))
        frame_idx += 1
    cap.release()

    if not frames:
        print("Không đọc được frame nào từ video.")
        return

    print(f"Đã đọc {len(frames)} frames.")

    # ── PHASE 1: Batch inference toàn bộ, đo tổng thời gian ──
    print(f"Running batch inference (batch_size={batch_size})...")
    all_results = []

    t_infer_start = time.time()
    for batch_start in tqdm.tqdm(range(0, len(frames), batch_size)):
        batch_chunk = frames[batch_start:batch_start + batch_size]

        batch_imgs0 = []
        batch_samples = []
        batch_ratios = []
        batch_pads = []
        batch_names = []

        for img_name, img0 in batch_chunk:
            batch_names.append(img_name)
            if img0 is None:
                batch_imgs0.append(None)
                batch_samples.append(None)
                batch_ratios.append(None)
                batch_pads.append(None)
                continue

            img, ratio, pad = resize(img0, 640, augment=False)
            sample = img.transpose((2, 0, 1))[::-1]
            sample = numpy.ascontiguousarray(sample)

            batch_imgs0.append(img0)
            batch_samples.append(sample)
            batch_ratios.append(ratio)
            batch_pads.append(pad)

        valid_indices = [i for i, s in enumerate(batch_samples) if s is not None]
        if not valid_indices:
            for img_name in batch_names:
                all_results.append((img_name, None, None, None, None))
            continue

        valid_samples = numpy.stack([batch_samples[i] for i in valid_indices])
        batch_tensor = torch.from_numpy(valid_samples).cuda().half() / 255.

        with torch.no_grad():
            outputs = model(batch_tensor)
            outputs = util.non_max_suppression(outputs, confidence_threshold=conf_thres, iou_threshold=iou_thres)

        valid_iter = iter(range(len(valid_indices)))
        for i, img_name in enumerate(batch_names):
            if i not in valid_indices:
                all_results.append((img_name, None, None, None, None))
                continue

            out_idx = next(valid_iter)
            det = outputs[out_idx]
            detections = det.cpu().numpy() if (det is not None and len(det)) else None
            all_results.append((img_name, batch_imgs0[i], detections, batch_ratios[i], batch_pads[i]))

    t_infer_end = time.time()
    total_infer_time = t_infer_end - t_infer_start
    infer_fps = len(frames) / total_infer_time
    print(f"Inference done: {len(frames)} frames in {total_infer_time:.2f}s → FPS: {infer_fps:.1f}")

    # ── PHASE 2: Vẽ box + FPS, lưu ảnh + tạo video ──
    print("Saving images and creating video...")
    first_img = next(img0 for _, img0, _, _, _ in all_results if img0 is not None)
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for img_name, img0, detections, ratio, pad in tqdm.tqdm(all_results):
        if img0 is None:
            continue

        h0, w0 = img0.shape[:2]

        if detections is not None:
            for x1, y1, x2, y2, conf, cls in detections:
                x1 = (x1 - pad[0]) / ratio[0]
                y1 = (y1 - pad[1]) / ratio[1]
                x2 = (x2 - pad[0]) / ratio[0]
                y2 = (y2 - pad[1]) / ratio[1]

                x1 = max(0, min(int(x1), w0))
                y1 = max(0, min(int(y1), h0))
                x2 = max(0, min(int(x2), w0))
                y2 = max(0, min(int(y2), h0))

                label = f'{names[int(cls)]} {conf:.2f}'
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(img0, f'FPS: {infer_fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, img_name), img0)
        out.write(img0)

    out.release()
    print(f"Video saved to {output_path}")

if __name__ == '__main__':
    main()