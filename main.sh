set -a
source "$(dirname "$0")/.env"
set +a

PYTHON="$(dirname "$0")/.venv/bin/python"

if [ ! -d "data/COCO" ]; then
    $PYTHON utils/prepare_dataset.py \
        --src  data/VisDrone2019-DET-train \
        --dst  data/COCO \
        --val-ratio 0 \
        --seed 42
fi

ARGS="--input-size $INPUT_SIZE --batch-size $BATCH_SIZE --epochs $EPOCHS --model $MODEL"
[ "$TRAIN" = "true" ] && ARGS="$ARGS --train"
[ "$TEST"  = "true" ] && ARGS="$ARGS --test"

if [ "$GPUS" -eq 1 ]; then
    $PYTHON main.py $ARGS
else
    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS main.py $ARGS
fi
