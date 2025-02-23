from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference, decode2frame

import os

video_path = "examples/sample_msrvtt.mp4"

# options ["ShareGPTVideo/LLaVA-Hound-DPO", "ShareGPTVideo/LLaVA-Hound-SFT", "ShareGPTVideo/LLaVA-Hound-SFT-Image_only"]
model_path = "/data2/wangxd/models/LLaVA-Hound-SFT" 
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, model_name=model_name, cache_dir=None)
inference_model = ModelInference(model=model, tokenizer=tokenizer, processor=processor, context_len=context_len)

# NOTE using frames can control the frame numer
# given 2 frames, just repeat to 8 frames

# our pipeline
# frame_dir, _ = os.path.splitext(video_path)
# decode2frame(video_path, frame_dir, verbose=True)

frame_dir = "/home/user/wangxd/LLaVA-Hound-DPO/llava_hound_dpo/examples/sample_msrvtt_2"

question="What is the evident theme in the video?"
response = inference_model.generate(
    question=question,
    modal_path=frame_dir,
    temperature=0,
)
print(response)

# using decord 
# response = inference_model.generate(
#     question=question,
#     modal_path=video_path,
#     temperature=0,
#     video_decode_backend="decord",
# )
# print(response)