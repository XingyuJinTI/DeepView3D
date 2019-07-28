#!/usr/bin/env bash

outdir=./output/tv_viewpoint_twice

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
    --batch_size 16 \
    --epoch_batches 640 \
    --eval_batches 2 \
    --optim adam \
    --lr 1e-3 \
    --epoch 200 \
    --vis_batches_vali 10 \
    --vis_every_train 10 \
    --gpu "$gpu" \
    --save_net 10 \
    --workers 8 \
    --logdir "$outdir" \
    --suffix '{classes}_canon-{canon_sup}' \
    --tensorboard \
    $*
source deactivate
