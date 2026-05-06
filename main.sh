GPUS=$1
PYTHON="$(dirname "$0")/.venv/bin/python"

if [ ! -d "data/COCO" ]; then
    $PYTHON utils/prepare_dataset.py \
        --src  data/VisDrone2019-DET-train \
        --dst  data/COCO \
        --val-ratio 0 \
        --seed 42
fi

if [ "$GPUS" -eq 1 ]; then
    $PYTHON main.py ${@:2}
else
    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS main.py ${@:2}
fi