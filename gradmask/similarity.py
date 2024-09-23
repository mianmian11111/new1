import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 加载两个 .npy 文件
file_path1 = f'result_log/sst2_bert/hidden_states/train12原始.npy'  # 替换为你的第一个文件路径
file_path2 = f'result_log/sst2_bert/hidden_states/train12微调后.npy'  # 替换为你的第二个文件路径

array1 = np.load(file_path1)  # 形状应该是 (13, n_sentences , 76, 768)
array2 = np.load(file_path2)  # 形状应该是 (13, n_sentences , 76, 768)

print("array1.shape:", array1.shape)
print("array2.shape:", array2.shape)

# 获取句子的数量
n_sentences = array1.shape[1]  # 假设第二维是句子数量

# 计算每个句子的每层平均表示，忽略第一个 token
avg_array1 = np.mean(array1[:, :, 1:, :], axis=2)  # 形状 (13, n_sentences , 768)
avg_array2 = np.mean(array2[:, :, 1:, :], axis=2)  # 形状 (13, n_sentences , 768)

# 存储相似度
similarity_matrix = np.zeros((n_sentences , 12))  # (1811, 12)
count =0 

# 遍历每个句子，计算相似度
for sentence in range(n_sentences ):
    for layer in range(12):
        similarity_matrix[sentence, layer] = cosine_similarity(
            avg_array1[layer:layer+1, sentence].reshape(1, -1),  # 第 layer 层的平均表示
            avg_array2[layer:layer+1, sentence].reshape(1, -1)   # 第 layer 层的平均表示
        )[0, 0]
    count = count+1

# 将每一层相似度转为字符串列表并保留四位小数
layers_sim = similarity_matrix.round(4).astype(str).tolist()  # 保留四位小数并转换为字符串列表

# 创建一个 DataFrame，并将 layers_sim 作为一列
df = pd.DataFrame({'layers_sim': layers_sim})

# 将 status 列添加到 DataFrame
df['status'] = True# 将状态列的值设为 True

# 将结果保存到 CSV 文件，只包含 layers_sim 和 status 列
df.to_csv('/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/dataset/sst2/sentence_layer_sim/bert_if_fine_tuning/train.csv', columns=['layers_sim', 'status'], index=False)

print("Results saved to sentence_layer_similarities.csv，num of sentence:", count )
