def add_tab_before_label(input_file, output_file):
    count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()  # 移除行首尾的空白字符
            if line:  # 确保行不是空的
                if line[-1] in ['0', '1']:
                    # 如果最后一个字符是0或1
                    new_line = line[:-1] + '\t' + line[-1] + '\n'
                    count = count + 1
                    outfile.write(new_line)
                else:
                    # 如果最后一个字符不是0或1，保持原样写入
                    outfile.write(line + '\n')
                    print("没有插入tab", line)
            else:
                # 如果是空行，直接写入
                outfile.write('\n')
                print("没有插入tab", line)
    print("tab number = ", count)

# 使用函数
input_file = 'dataset/sst2/att01.txt'
output_file = 'dataset/sst2/att02.txt'
add_tab_before_label(input_file, output_file)
print(f"处理完成，结果保存在 {output_file}")