input_model_name=${1:-"/data2/wangxd/models/LLaVA-Hound-SFT"}
lr=${3:-"5e-7"}

CACHE_DIR=/data2/wangxd/.cache
cache_dir=$CACHE_DIR
export cache_dir=$cache_dir

# export WANDB_MODE=disabled
export WANDB_PROJECT=llava-hound-sft
export WANDB_NAME=LLaVA-Hound-SFT-debate-SimPO-17k_top_p1.0_temp1.2-ls0.1

# gpu_ids=0
gpu_ids=4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

model_name_or_path=$input_model_name
output_dir=/data2/wangxd/ckpt/${WANDB_PROJECT}/${WANDB_NAME}
mkdir -p $output_dir

# DATA debate data
data_path=/home/user/wangxd/LLaVA-Hound-DPO/llava_hound_dpo/self-gen/LLaVA-Hound-SFT_debate_aug_17k_top_p1.0_temp1.2.jsonl

video_dir=/home/user/wangxd/LLaVA-NeXT/data/shareVideoGPTV/dpo_train_data
image_dir="/"

# sudo chmod +x -R .
export PYTHONPATH=.
rand=$RANDOM
port=$((19000 + $rand % 1000))

torchrun --nproc_per_node=$n_gpu --master_port=$port dpo_scripts/run_dpo_avg.py \
    --deepspeed config/zero2.json \
    --model_name_or_path $model_name_or_path \
    --loss_type simpo \
    --label_smoothing 0.1 \
    --dpo_alpha 1.0 --beta 2.0 --gamma 0.5 \
    --version v1 \
    --data_path $data_path \
    --video_folder $video_dir \
    --image_folder $image_dir \
    --X "Image" "Video" --training_modal 'video' \
    --image_tower /data2/wangxd/models/LanguageBind/LanguageBind_Image \
    --video_tower /data2/wangxd/models/LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_only_model True \
    --save_total_limit 10 \
    --learning_rate $lr --freeze_mm_mlp_adapter True \
    --weight_decay 0. --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --cache_dir $cache_dir \
    --report_to wandb 2>&1 | tee $output_dir/train.log