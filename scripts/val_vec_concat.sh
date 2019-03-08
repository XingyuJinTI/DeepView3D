#!/usr/bin/env bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-72:00:00
#SBATCH --exclude=landonia23

outdir=./output/tvmarrnet_vec_concat_val
trained_model=./downloads/models/tvmarrnet_vec_concat.pt

rm -rf $outdir

export STUDENT_ID=$(whoami)

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
pred_thresh=0.3
if [ $# -ge 3 ]; then
    pred_thresh=$3
fi
gpu="$1"
class="$2"
shift # shift the remaining arguments
shift

set -e

source /home/${STUDENT_ID}/miniconda3/bin/activate shaperecon

python validate.py \
    --net tvmarrnet_vec_concat \
    --dataset shapenet2 \
    --pred_thresh $pred_thresh \
    --classes "$class" \
    --canon_sup \
    --trained_model "$trained_model"\
    --batch_size 4 \
    --epoch_batches 2500 \
    --eval_batches 5 \
    --optim adam \
    --lr 1e-3 \
    --epoch 1000 \
    --vis_batches_vali 10 \
    --gpu "$gpu" \
    --save_net 10 \
    --workers 4 \
    --logdir "$outdir" \
    --suffix '{classes}_canon-{canon_sup}' \
    --tensorboard \
    $*

source /home/${STUDENT_ID}/miniconda3/bin/deactivate
