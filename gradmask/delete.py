def remove_empty_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():  # 如果行不为空（包括只有空格的行）
                outfile.write(line)

# 使用函数
input_file = 'dataset/sst2/att01.txt'
output_file = 'dataset/sst2/att02.txt'
remove_empty_lines(input_file, output_file)
print("空行已删除，结果保存在", output_file)