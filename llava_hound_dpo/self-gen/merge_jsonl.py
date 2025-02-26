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

file_list = ['LLaVA-Hound-SFT_aug_only_f1_0_4250_top_p1.0_temp1.2.jsonl', 'LLaVA-Hound-SFT_aug_only_f1_4250_8500_top_p1.0_temp1.2.jsonl', 'LLaVA-Hound-SFT_aug_only_f1_8500_12750_top_p1.0_temp1.2.jsonl', 'LLaVA-Hound-SFT_aug_only_f1_12750_17000_top_p1.0_temp1.2.jsonl']


output_file = 'LLaVA-Hound-SFT_aug_only_f1_17k_top_p1.0_temp1.2.jsonl'


merge_jsonl_files(file_list, output_file)

print(f"文件已合并到 {output_file}")