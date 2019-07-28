#!/usr/bin/env bash

outdir=./output/marrnet2_std_gpu

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
gpu="$1"
class="$2"
shift # shift the remaining arguments
shift

rm -rf ${outdir}/marrnet2_shapenet_0.001_chair_canon-True/0/tensorboard

set -e

#export FILE_CHECK1=/disk/scratch/${TEAM_NAME}/data/shapenet/03001627/ue639c33f-d415-458c-8ff8-2ef68135af15/03001627_ue639c33f-d415-458c-8ff8-2ef68135af15_voxel_normalized_128.mat

#if [ -f "$FILE_CHECK1" ]; then
#    echo "ShapeNet files exist"
#else
#    echo "Copying ShapeNet files"
#    mkdir -p /disk/scratch/${TEAM_NAME}/data/shapenet/03001627/
#    rsync -ua --progress /home/${STUDENT_ID}/GenRe-ShapeHD/downloads/data/shapenet/03001627/ /disk/scratch/${TEAM_NAME}/data/shapenet/03001627/
#fi

#export FILE_CHECK2=/disk/scratch/${TEAM_NAME}/data/shapenet/status/vox_rot.txt

#if [ -f "$FILE_CHECK2" ]; then
#    echo "Status files exist"
#else
#    echo "Copying status files"
#    mkdir -p /disk/scratch/${TEAM_NAME}/data/shapenet/status/
#    rsync -ua --progress /home/${STUDENT_ID}/GenRe-ShapeHD/downloads/data/shapenet/status/ /disk/scratch/${TEAM_NAME}/data/shapenet/status/
#fi

source /home/${STUDENT_ID}/miniconda3/bin/activate shaperecon

python train.py \
    --net marrnet2 \
    --dataset shapenet \
    --classes "$class" \
    --canon_sup \
    --batch_size 4 \
    --epoch_batches 2500 \
    --eval_batches 5 \
    --optim adam \
    --lr 1e-3 \
    --epoch 9 \
    --vis_batches_vali 10 \
    --gpu "$gpu" \
    --save_net 10 \
    --workers 4 \
    --logdir "$outdir" \
    --suffix '{classes}_canon-{canon_sup}' \
    --tensorboard \
    --resume -1 \
    $*

source /home/${STUDENT_ID}/miniconda3/bin/deactivate
