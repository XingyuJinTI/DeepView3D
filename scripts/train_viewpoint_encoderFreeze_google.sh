#!/usr/bin/env bash

outdir=./output/tv_viewpoint_freeze_encoder

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
gpu="$1"
class="$2"
shift # shift the remaining arguments
shift

set -e
rm ./output/tv_viewpoint_freeze_encoder/tvmarrnet_vp_freezeEncoder_shapenet2_0.001_chair_canon-True/0/tensorboard
source activate shaperecon

python train.py \
    --net tvmarrnet_vp_freezeEncoder \
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
    --resume -1 \
    $*
source deactivate
