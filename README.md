YOLOv11 re-implementation using PyTorch

### Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv .venv --python 3.10
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

### Dataset

Place VisDrone2019-DET dataset under `data/` then convert to YOLO format:

```bash
python utils/prepare_dataset.py \
    --src data/VisDrone2019-DET-train \
    --dst data/COCO \
    --val-ratio 0.1 \
    --seed 42
```

### Train & Test

All config is in `.env`. Edit it, then run:

```bash
bash main.sh
```

Key parameters in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GPUS` | `1` | Number of GPUs |
| `MODEL` | `yolo_v11_s` | Model variant: `yolo_v11_n/t/s/m/l/x`, `hierlight_yolo_n/s/m` |
| `EPOCHS` | `600` | Training epochs |
| `BATCH_SIZE` | `16` | Batch size |
| `INPUT_SIZE` | `640` | Input image size |
| `TRAIN` | `true` | Run training |
| `TEST` | `false` | Run evaluation after training |
| `WEIGHTS` | ` ` | Pretrained backbone (e.g., `backbone/v11_n.pt`) hoặc checkpoint để resume (e.g., `weights/last.pt`) |

### Resume training

Nếu training bị crash, resume từ checkpoint cuối:

```bash
# Trong .env
WEIGHTS=weights/last.pt
```

Sau đó chạy lại (phải đứng trong thư mục `train_yolov11/`):

```bash
cd train_yolov11 && ./main.sh
```

Checkpoint `last.pt` lưu model (EMA), optimizer, amp scaler, best mAP và epoch — training sẽ tiếp tục đúng từ epoch bị crash, learning rate scheduler tự khớp theo global step.

### Results

| Version | Epochs | Box mAP |                                                                              Download |
|:-------:|:------:|--------:|--------------------------------------------------------------------------------------:|
|  v11_n  |  600   |    38.6 |                                                            [Model](./weights/best.pt) |
| v11_n*  |   -    |    39.2 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_n.pt) |
| v11_s*  |   -    |    46.5 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_s.pt) |
| v11_m*  |   -    |    51.2 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_m.pt) |
| v11_l*  |   -    |    53.0 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_l.pt) |
| v11_x*  |   -    |    54.3 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_x.pt) |

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.551
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.777
```

* `*` means that it is from original repository, see reference
* In the official YOLOv11 code, mask annotation information is used, which leads to higher performance

### Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

#### Reference

* https://github.com/ultralytics/ultralytics
* https://github.com/jahongir7174/YOLOv8-pt
