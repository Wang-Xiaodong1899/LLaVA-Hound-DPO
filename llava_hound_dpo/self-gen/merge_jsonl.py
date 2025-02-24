import json

def merge_jsonl_files(file_list, output_file):
    """
    按顺序合并多个 jsonl 文件到一个新的 jsonl 文件

    :param file_list: 文件名列表，按顺序合并
    :param output_file: 输出的合并后的文件名
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in file_list:
            with open(file_name, 'r', encoding='utf-8') as infile:
                for line in infile:
                    
                    outfile.write(line)

file_list = ['LLaVA-Hound-SFT_debate_0_4000_temp0.jsonl', 'LLaVA-Hound-SFT_debate_4000_8000_temp0.jsonl', 'LLaVA-Hound-SFT_debate_8000_12000_temp0.jsonl', 'LLaVA-Hound-SFT_debate_12000_17000_temp0.jsonl']


output_file = 'LLaVA-Hound-SFT_debate_17k_temp0.jsonl'


merge_jsonl_files(file_list, output_file)

print(f"文件已合并到 {output_file}")