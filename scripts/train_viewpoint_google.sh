#!/usr/bin/env bash

outdir=./output/tv_viewpoint

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
gpu="$1"
class="$2"
shift # shift the remaining arguments
shift

set -e

source activate shaperecon

python train.py \
    --net tvmarrnet_vp \
    --dataset shapenet2 \
    --classes "$class" \
    --canon_sup \
    --batch_size 4 \
    --epoch_batches 2500 \
    --eval_batches 5 \
    --optim adam \
    --lr 1e-3 \
    --epoch 300 \
    --vis_batches_vali 10 \
    --gpu "$gpu" \
    --save_net 10 \
    --workers 10 \
    --logdir "$outdir" \
    --suffix '{classes}_canon-{canon_sup}' \
    --tensorboard \
    # --resume -1 \
    $*
source deactivate
