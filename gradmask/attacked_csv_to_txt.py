import pandas as pd

# 读取 CSV 文件
csv_file = 'result_log/sst2_bert/attack-len128-epo10-batch32/attack_results.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(csv_file)

# 筛选出 result_type 为 "Successful" 的行
successful_attacks = df[df['result_type'] == 'Successful']

# 提取 perturbed_text 和 ground_truth_output 列
perturbed_texts = successful_attacks['perturbed_text']
ground_truth_outputs = successful_attacks['ground_truth_output'].astype(int)

# 将结果保存到 TXT 文件中
txt_file = 'dataset/sst2/attacked_succ.txt'  # 输出的 TXT 文件路径
with open(txt_file, 'w') as f:
    for text, output in zip(perturbed_texts, ground_truth_outputs):
        f.write(f"{text}\t{output}\n")

print(f"Successfully saved perturbed_text and ground_truth_output to {txt_file}")