import numpy as np
from scipy.spatial.distance import cosine
# 可视化
import matplotlib.pyplot as plt
import warnings

def safe_cosine(u, v):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.all(u == 0) or np.all(v == 0):
            return 0.0  # 如果任一向量为零向量，返回0相似度
        return cosine(u, v)



def compute_layer_similarity(hidden_states):
    num_layers, num_batches, seq_length, hidden_size = hidden_states.shape
    similarities = []
    
    # 第 0 层
    layer_0 = hidden_states[0]  # 形状为 (n, m, 768)
    
    # 计算与后面每一层的相似度
    for i in range(1, num_layers):
        layer_i = hidden_states[i]  # 形状为 (n, m, 768)
        
        batch_similarities = []
        for b in range(num_batches):
            token_similarities = []
            for t in range(seq_length):
                # 计算每个 token 的余弦相似度
                similarity = 1 - safe_cosine(layer_0[b, t], layer_i[b, t])
                token_similarities.append(similarity)
            # 计算这个批次的平均相似度
            batch_similarities.append(np.mean(token_similarities))
        
        # 计算所有批次的平均相似度
        avg_similarity = np.round(np.mean(batch_similarities), decimals=2)
        similarities.append(avg_similarity)
    
    return similarities


# 指定你的 .npy 文件路径
file_path1 = 'result_log/sst2_bert/hidden_stages/cln.npy'
file_path2 = 'result_log/sst2_bert/hidden_stages/att.npy'

# 使用 np.load 加载 .npy 文件
hidden_states_cln = np.load(file_path1)
hidden_states_att = np.load(file_path2)
# 使用方法
# 假设 hidden_states 是您的 (13, n, m, 768) 数组
similarities_cln = compute_layer_similarity(hidden_states_cln)
similarities_att = compute_layer_similarity(hidden_states_att)



print(f"Similarity_att between Layer 0 and Layer :", similarities_att)
print(f"Similarity_cln between Layer 0 and Layer :", similarities_cln)