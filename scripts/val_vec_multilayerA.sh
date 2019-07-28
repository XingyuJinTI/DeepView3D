#!/usr/bin/env bash

trained_model=./downloads/models/tvmarrnet_vec_multilayerA.pt


if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
gpu="$1"
class="$2"

outdir=./output/tvmarrnet_vec_multilayerA_val
pred_thresh=0.3
if [ $# -ge 3 ]; then
    pred_thresh="$3"
    outdir=./output/tvmarrnet_vec_multilayerA_val_${pred_thresh}
    echo "pred_thresh: $pred_thresh"
    echo "$outdir"
fi

shift # shift the remaining arguments
shift
shift

rm -rf $outdir

set -e

source activate shaperecon

python validate.py \
    --net tvmarrnet_vec_multilayerA \
    --dataset shapenet2 \
    --pred_thresh "$pred_thresh" \
    --classes "$class" \
    --canon_sup \
    --trained_model "$trained_model"\
    --batch_size 4 \
    --epoch_batches 2500 \
    --eval_batches 5 \
    --optim adam \
    --lr 1e-3 \
    --epoch 1 \
    --vis_batches_vali 10 \
    --gpu "$gpu" \
    --save_net 10 \
    --workers 4 \
    --logdir "$outdir" \
    --suffix '{classes}_canon-{canon_sup}' \
    --tensorboard \
    $*

source deactivate
