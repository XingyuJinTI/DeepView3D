#!/usr/bin/env bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:2
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-8:00:00
#SBATCH --exclude=landonia23

outdir=./output/marrnet2_gpu_val
trained_model=./output/marrnet2_std_gpu/marrnet2_shapenet_0.001_chair_canon-True/0/best.pt

rm -rf $outdir

export STUDENT_ID=$(whoami)

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
gpu="$1"
class="$2"
shift # shift the remaining arguments
shift

set -e

source /home/${STUDENT_ID}/miniconda3/bin/activate shaperecon

python validate.py \
    --net marrnet2 \
    --dataset shapenet \
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
