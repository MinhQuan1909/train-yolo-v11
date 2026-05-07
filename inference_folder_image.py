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
    # ── CONFIG ──
    device = 'cpu'  # 'cuda' hoặc 'cpu'
    image_dir = 'data_val/SOT_video1_uav0000024_00000_s'
    output_dir = f"{image_dir}_result"
    os.makedirs(output_dir, exist_ok=True)

    conf_thres = 0.1
    iou_thres = 0.2
    fps = 30
    output_path = f'output_video-confidence_threshold_{conf_thres}-iou_threshold_{iou_thres}.mp4'
    batch_size = 64

    # Load model
    print("Loading model...")
    model = torch.load('./weights/best.pt', map_location=device, weights_only=False)
    if device == 'cuda':
        model = model['model'].float().fuse().half().eval()
    else:
        model = model['model'].float().fuse().eval()

    # Load class names
    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
    names = params['names']

    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort()

    if not images:
        print("No images found in", image_dir)
        return

    print(f"Found {len(images)} images.")

    # ── PHASE 1: Batch inference ──
    print(f"Running batch inference (batch_size={batch_size})...")
    all_results = []

    t_infer_start = time.time()
    for batch_start in tqdm.tqdm(range(0, len(images), batch_size)):
        batch_names = images[batch_start:batch_start + batch_size]

        batch_imgs0 = []
        batch_samples = []
        batch_ratios = []
        batch_pads = []

        for img_name in batch_names:
            img_path = os.path.join(image_dir, img_name)
            img0 = cv2.imread(img_path)
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
        batch_tensor = torch.from_numpy(valid_samples).to(device)
        if device == 'cuda':
            batch_tensor = batch_tensor.half() / 255.
        else:
            batch_tensor = batch_tensor.float() / 255.

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
    infer_fps = len(images) / total_infer_time
    print(f"Inference done: {len(images)} frames in {total_infer_time:.2f}s → FPS: {infer_fps:.1f}")

    # ── PHASE 2: Vẽ box + lưu ảnh + tạo video ──
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