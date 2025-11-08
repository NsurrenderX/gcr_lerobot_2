#!/bin/bash

# 默认参数值
NNODES=1
NPROC_PER_NODE=2
JOB_NAME=""
JOB_TYPE="pretrain"
DATA_MIX="oxe_magic_soup_plus"
BATCH_SIZE=32
OPTIMIZER_LR=2.5e-5
OPTIMIZER_DECAY_LR=2.5e-6
OPTIMIZER_WEIGHT_DECAY=1e-2
SCHEDULER_WARMUP_STEPS=1000
SCHEDULER_DECAY_STEPS=30000
SAVE_FREQ=2000
SCHEDULER_PLATFORM_STEPS=1
PRETRAINED_PATH=""
GRADIENT_ACCUMULATION_STEPS=4
FREEZE_VISION="false"
TRAIN_FULL_VLM='true'
TRAIN_AWA='true'
TRAIN_EXPERT='true'
IMAGE_AUG='true'

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --data_mix)
            DATA_MIX="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --job_name)
            JOB_NAME="$2"
            shift 2
            ;;
        --job_type)
            JOB_TYPE="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --optimizer_lr)
            OPTIMIZER_LR="$2"
            shift 2
            ;;
        --weight_decay)
            OPTIMIZER_WEIGHT_DECAY="$2"
            shift 2
            ;;
        --full_vlm)
            TRAIN_FULL_VLM="$2"
            shift 2
            ;;
        --expert)
            TRAIN_EXPERT="$2"
            shift 2
            ;;
        --awa)
            TRAIN_AWA="$2"
            shift 2
            ;;
        --freeze_vision)
            FREEZE_VISION="$2"
            shift 2
            ;;
        --aug)
            IMAGE_AUG="$2"
            shift 2
            ;;
        --scheduler_decay_lr)
            OPTIMIZER_DECAY_LR="$2"
            shift 2
            ;;
        --scheduler_warmup_steps)
            SCHEDULER_WARMUP_STEPS="$2"
            shift 2
            ;;
        --scheduler_decay_steps)
            SCHEDULER_DECAY_STEPS="$2"
            shift 2
            ;;
        --scheduler_platform_steps)
            SCHEDULER_PLATFORM_STEPS="$2"
            shift 2
            ;;
        --pre_path)
            PRETRAINED_PATH="$2"
            shift 2
            ;;
        --grad_acc)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [[ -z "$JOB_NAME" ]]; then
    echo "错误：必须指定 --job_name"
    exit 1
fi

# 固定输出目录（根据需求修改）
FIXED_OUTPUT_DIR="/mnt/wangxiaofa/original_qw"

# 执行训练命令
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    lerobot/scripts/fsdp_train.py \
    --policy.type="qwen" \
    --output_dir="$FIXED_OUTPUT_DIR" \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=$IMAGE_AUG \
    --batch_size=$BATCH_SIZE \
    --log_freq=100 \
    --save_freq=$SAVE_FREQ \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --data_mix=$DATA_MIX \
    --dataset.processor="/mnt/wangxiaofa/qwen_params/Qwen2.5-VL-7B-Instruct/" \
    --dataset.parent_dir="/scratch/amlt_code/gcr_lerobot_2/robot_dataset/lerobot-format/" \
    --policy.scheduler_warmup_steps=$SCHEDULER_WARMUP_STEPS \
    --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
    --policy.scheduler_platform_steps=$SCHEDULER_PLATFORM_STEPS \
    --policy.optimizer_lr=$OPTIMIZER_LR \
    --policy.optimizer_weight_decay=$OPTIMIZER_WEIGHT_DECAY \
    --policy.scheduler_decay_lr=$OPTIMIZER_DECAY_LR \
    --policy.freeze_vision_encoder=$FREEZE_VISION \
    --policy.train_expert_only=false \
    --policy.train_awa=$TRAIN_AWA \
    --policy.train_expert=$TRAIN_EXPERT \
    --policy.train_full_vlm=$TRAIN_FULL_VLM \
    --resume=true \
    --wandb.enable=true \
    --wandb.project="pi0first" \
    --job_name="$JOB_NAME" \
    --job_type="$JOB_TYPE" \
    --log_dir="/mnt/wangxiaofa/logs" \
    --policy.pretrained_path=$PRETRAINED_PATH