import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine

# 指定你的 .npy 文件路径
file_path1 = 'result_log/sst2_bert/hidden_stages/cln0.npy'
file_path2 = 'result_log/sst2_bert/hidden_stages/cln.npy'

# 使用 np.load 加载 .npy 文件
array1 = np.load(file_path1)
array2 = np.load(file_path2)
    
print("array2.shape:", array2.shape)
print("array1.shape:", array1.shape)

    
# 初始化结果数组
layer_similarities = np.zeros(13)
    
# 遍历每一层
for layer in range(13):
    sentence_similarities = np.zeros(len(array1[0]))
        
    # 遍历每个句子
    for sentence in range(len(array1[0])):
        a = array1[layer, sentence].reshape(-1)
        b = array2[layer, sentence].reshape(-1)
        if(len(a) > len(b)):
            a = a[:len(b)]
        else:
            b = b[:len(a)]
        # 计算当前(53, 768)子数组的相似度
        similarity = 1-cosine(a, b)
        sentence_similarities[sentence] = similarity
        
    # 计算这一层所有句子的平均相似度
    layer_similarities[layer] = np.mean(sentence_similarities)
print(layer_similarities)