#!/bin/bash
ROOT_DIR="/workspace/wxd/LLaVA-Hound-DPO/llava_hound_dpo"

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=$1
FRAMES=$2
OVERWRITE=$3
SAVE_NAME=$4
DURATION=$5

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_frames_${FRAMES}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_frames_${FRAMES}
fi

echo $RESOLUTION
    
python3 test/eval/video_mme.py \
    --model-path $CKPT \
    --output_dir ./work_dirs/video_demo/$SAVE_DIR \
    --output_name test \
    --overwrite ${OVERWRITE} \
    --for_get_frames_num $FRAMES \
    --answers-file results/answer-video-mme-${SAVE_NAME}.json \
    --duration $DURATION \
