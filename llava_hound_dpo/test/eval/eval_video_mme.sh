#!/bin/bash

CKPT=$1
SAVE_NAME=$2
FRAMES=$3
DURATION=$4 # short
OVERWRITE=$5 # overwrite previous eval result

#eval_frame: 16 (align with finetuning)
if [ "$OVERWRITE" = True ]; then
    bash test/eval/video_mme.sh $CKPT $FRAMES True $SAVE_NAME $DURATION
fi

python3 test/eval/eval_video_mme.py \
    --results_file results/answer-video-mme-${SAVE_NAME}.json  \
    --video_duration_type $DURATION \
    --return_categories_accuracy

# tip
# vicuna: llava-hound-dpo
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=3 bash test/eval/eval_video_mme.sh /volsparse3/wxd/models/vicuna/LLaVA-Hound-SFT LLaVA-Hound-SFT-short 8 short True
