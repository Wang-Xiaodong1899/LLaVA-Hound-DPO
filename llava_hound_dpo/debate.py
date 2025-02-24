from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference, decode2frame

import fire
import os
import json
from tqdm import tqdm
import torch


def run_inference(start=0, end=17000):
    model_path = "/data2/wangxd/models/LLaVA-Hound-SFT" 
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, model_name=model_name, cache_dir=None)
    inference_model = ModelInference(model=model, tokenizer=tokenizer, processor=processor, context_len=context_len)


    output_dir = "self-gen"
    os.system(f"mkdir -p {output_dir}")


    output_name = f"LLaVA-Hound-SFT_debate_{start}_{end}"
    answers_file = os.path.join(output_dir, f"{output_name}.jsonl")
    ans_file = open(answers_file, "w")

    video_root = "/home/user/wangxd/LLaVA-NeXT/data/shareVideoGPTV/dpo_train_data"
    
    # XXX input
    jsonl_file = "/home/user/wangxd/LLaVA-NeXT/data/shareVideoGPTV/sft_dpo_17k.jsonl"
    

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        jsonl_data = [json.loads(line) for line in file]

    # import pdb;pdb.set_trace()
    for item in tqdm(jsonl_data[start:end]):
        sample_set = {}
        video_ = item["video"]
        sample_set['id'] = item["id"]
        
        
        answer = item["answer"]
        
        question = item["prompt"]
        
        question = question.replace("<video>", "").replace("\n", "")

        sample_set["prompt"] = question
        sample_set["answer"] = answer
        
        sample_set["video"] = video_
        
        
        video_path = os.path.join(video_root, video_) # so many frames
        
        video = None
        
        K = 2
        # import pdb; pdb.set_trace()
        model_return = None
        outputs_list = []
        for turn in range(K):
            qs = question
            # TODO inject GT info
            if turn == 0:
                prefix = "Here are some hints: " + answer + "\n\n" + "Please respond based on the given hints and video content." + "\n\n"
            else:
                prefix = f"""
Your previous reply to me was:
{model_return}. This response can continue to be improved.

Now, please align your response with the information below:
{answer}

You need to reflect the given information as best you can, optimize your response, and enrich your answer. I'll ask you the question again:

"""
            with torch.no_grad():
                outputs = inference_model.generate(
                    question=prefix + question,
                    modal_path=video_path, # only take 8 frames
                    temperature=0, # if temperature < 0.01== do_sample
                    top_p=0.9,
                )
                # print(outputs)
                
                outputs = outputs.strip()
            
                previous_return = model_return if model_return else ""
                model_return = outputs
            
        sample_set["first"] = previous_return
        sample_set["chosen"] = model_return
        # sample_set["rejected"] = outputs_list[1]
        
        ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    fire.Fire(run_inference)