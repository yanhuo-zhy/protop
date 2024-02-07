#!/bin/bash 
#SBATCH --account cvl
#SBATCH -p amp48
#SBATCH --qos amp48
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/protop/temp/baseline_supcon.txt

module load gcc/gcc-10.2.0
# module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/pszzz/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python main.py \
    --base_architecture='deit_base_patch16_224' \
    --data_set='CD_CUB2011U' \
    --data_path='datasets' \
    --input_size=224 \
    --output_dir='output_cosine/CD_CUB2011U/baseline_supcon' \
    --model='deit_base_patch16_224' \
    --batch_size=128 \
    --seed=1028 \
    --opt='adamw' \
    --sched='cosine' \
    --warmup-epochs=5 \
    --warmup-lr=1e-4 \
    --decay-epochs=10 \
    --decay-rate=0.1 \
    --weight_decay=0.05 \
    --epochs=200 \
    --finetune='protopformer' \
    --features_lr=1e-4 \
    --add_on_layers_lr=3e-3 \
    --prototype_vectors_lr=3e-3 \
    --prototype_shape 1000 192 1 1 \
    --reserve_layers 11 \
    --reserve_token_nums 196 \
    --use_global=True \
    --use_ppc_loss=False \
    --ppc_cov_thresh=1.  \
    --ppc_mean_thresh=2. \
    --global_coe=0.5 \
    --global_proto_per_class=10 \
    --ppc_cov_coe=0.1 \
    --ppc_mean_coe=0.5