import random

def shuffle_file(input_file, output_file):
    # 读取所有行
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 打乱行的顺序
    random.shuffle(lines)
    
    # 写入打乱后的行
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# 使用示例
input_file = '/dataset/sst2/train.txt'
output_file = '/dataset/sst2/train_shuffled.txt'
shuffle_file(input_file, output_file)
print(f"文件已打乱并保存为 {output_file}")
