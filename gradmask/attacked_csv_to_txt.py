import csv

# CSV 文件路径
csv_file_path = '/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/result_log/sst2_bert/attack-len128-epo10-batch32/deepwordbug/test/attack_results.csv'
# TXT 文件路径
txt_file_path = '/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/dataset/sst2/att_test.txt'

# 打开 CSV 文件并读取数据
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    # 创建 CSV 读取器
    csv_reader = csv.DictReader(csv_file)
    
    # 打开 TXT 文件准备写入
    with open(txt_file_path, mode='w', encoding='utf-8') as txt_file:
        # 遍历 CSV 文件中的每一行
        for row in csv_reader:
            # 提取 'perturbed_text' 和 'original_output' 列
            perturbed_text = row['perturbed_text']
            original_output = row['original_output']
            # 删除 perturbed_text 中的 '[[', ']]'
            perturbed_text = perturbed_text.replace('[[', '').replace(']]', '')
            
            # 将它们用制表符分隔后写入 TXT 文件，每个样本一行
            txt_file.write(f"{perturbed_text}\t{original_output}\n")
