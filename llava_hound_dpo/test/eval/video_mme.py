import argparse
import datasets as hf_datasets
import json
import os
import sys
import torch

sys.path.append("/home/user/wangxd/LLaVA-Hound-DPO/llava_hound_dpo")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference, decode2frame

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
import openai

from PIL import Image

import shortuuid

import numpy as np

MAX_IMAGE_LENGTH = 16

def _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=336, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.
    video_mask = np.zeros(max_frames, dtype=np.int64)
    max_video_length = 0

    # T x 3 x H x W
    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        
        # print(f'all pos {len(all_pos)}')
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
        slice_len = patch_images.shape[0]
        
        # print(f'slice_len: {slice_len}')

        max_video_length = max_video_length if max_video_length > slice_len else slice_len
        if slice_len < 1:
            pass
        else:
            video[:slice_len, ...] = patch_images
            # print(f"patch_images size: {patch_images.size()}")
        # print(f'len of patch_images: {len(patch_images)}')
        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))

    video_mask[:max_video_length] = [1] * max_video_length

    return torch.from_numpy(video), video_mask


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="/data2/wangxd/models/LLaVA-Hound-SFT")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
 
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=8)

    parser.add_argument("--video-folder", type=str, default="/home/user/wangxd/LLaVA-NeXT/data/Video-MME/data")
    parser.add_argument("--question-file", type=str, default="/home/user/wangxd/LLaVA-NeXT/llava/eval/questions/video_qa/temporal_qa.json")
    parser.add_argument("--answers-file", type=str, default="results/answer-video-mme.json")
    parser.add_argument("--duration", type=str, default="short")
    parser.add_argument("--subtitle", action="store_true") # TODO

    
    
    return parser.parse_args()


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    model_path = args.model_path
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, model_name=model_name, cache_dir=None)
    inference_model = ModelInference(model=model, tokenizer=tokenizer, processor=processor, context_len=context_len)

    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    hf_data = hf_datasets.load_dataset("parquet", data_files="/home/user/wangxd/LLaVA-NeXT/data/Video-MME/test-00000-of-00001.parquet")['train']
    keys = ['video_id', 'duration', 'domain', 'sub_category', 'url', 'videoID', 'question_id', 'task_type', 'question', 'options', 'answer']

    save_data = []
    groups = {}
    

    # generate answer by order
    for idx in tqdm(range(len(hf_data))):
        sample = hf_data[idx]
        if sample["duration"] != args.duration:
            continue
        
        video_num = sample["video_id"] # eg. 001
        video_name = sample["videoID"] # eg. fFjv93ACGo8
        question_id = sample["question_id"]
        question = sample["question"]
        options = sample["options"] # eg. ["A. xxx", "B. xxx", ...]
        answer = sample["answer"]
        
        
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break
        
        qs = f"""
        Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
        {question}
        {options[0]}
        {options[1]}
        {options[2]}
        {options[3]}
        The best answer is:
        """
        
        # print(qs)
        try:
            # using decord 
            response = inference_model.generate(
                question=qs,
                modal_path=video_path,
                temperature=0,
                video_decode_backend="decord",
            )
            
            # save answer by video_num
            current_response = {
                "question_id": sample["question_id"],
                "task_type": sample["task_type"],
                "question": sample["question"],
                "options": sample["options"],
                "answer": sample["answer"],
                "response": response, 
            }
            if video_num not in groups:
                groups[video_num] = {
                    "video_id": video_num,
                    "duration": sample["duration"],
                    "domain": sample["domain"],
                    "sub_category": sample["sub_category"],
                    "questions": [current_response]
                }
            else:
                groups[video_num]["questions"].append(current_response)
            
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    for key in groups.keys():
        save_data.append(groups[key])
    
    ans_file = open(answers_file, "w")
    json.dump(save_data, ans_file)
    ans_file.close()
    

if __name__ == "__main__":
    args = parse_args()
    print(f'eval frames: {args.for_get_frames_num}')
    run_inference(args)
