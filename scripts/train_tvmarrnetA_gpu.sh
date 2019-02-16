#!/usr/bin/env bash
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-72:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export TEAM_NAME='teebeedee'

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

outdir=./output/tvmarrnetA_gpu

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
gpu="$1"
class="$2"
shift # shift the remaining arguments
shift

set -e

mkdir -p /disk/scratch/${TEAM_NAME}/data/shapenet/03001627/
mkdir -p /disk/scratch/${TEAM_NAME}/data/shapenet/status/
rsync -ua --progress /home/${STUDENT_ID}/GenRe-ShapeHD/downloads/data/shapenet/03001627/ /disk/scratch/${TEAM_NAME}/data/shapenet/03001627/
rsync -ua --progress /home/${STUDENT_ID}/GenRe-ShapeHD/downloads/data/shapenet/status/ /disk/scratch/${TEAM_NAME}/data/shapenet/status/
export DATASET_DIR = /disk/scratch/${TEAM_NAME}/data/shapenet/

source /home/${STUDENT_ID}/miniconda3/bin/activate shaperecon

python train.py \
    --net tvmarrnetA \
    --dataset shapenet2 \
    --classes "$class" \
    --canon_sup \
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
