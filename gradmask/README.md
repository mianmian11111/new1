## 相似度检测方法：

#### 1.微调的bert：只用一个分类器进行训练，用**第12层**的隐藏层向量作为这个分类器的输入，得到的模型有两个部分：一个是encoder_调，一个是mlp_调。	

> ```
> python main.py --mode train --dataset_name sst2
> ```
>
> 加载训练之后的模型参数，accuracy test: 0.9227

#### 2.原始的bert：一个**原始的bert**，没有微调。模型只有一个部分：encoder_原。

> 不加载模型参数，accuracy test: 0.4975

#### 3.攻击test和train数据，得到攻击成功的数据

```
1.textattack attack ...

# 处理攻击后的csv数据，得到att_test\att_train\org_test\org_train
2.python main.py --mode attack_csv_to_txt --dataset_name sst2
```

#### 4.对**干净的训练样本**，放入‘1’用第12层训练之后的bert模型和原始的bert模型， 对于每一层计算他们的隐藏层向量的相似度。 所以一个干净的样本将得到一个12维的相似度。

> accuracy test:微调前：0.4793，微调后：0.9994（保存成csv
>
> layers_sim,status
> "['0.9975', '0.9756', '0.9493', '0.8889', '0.8417', '0.7948', '0.7617', '0.6817', '0.6156', '0.4394', '0.2895', '0.2865']",True
>
> ```
> 1.
> # 输出隐藏层，默认hidden_data_type = ['att_test', 'att_train', 'org_test','org_train'] 
> python main.py --mode output_hidden --dataset_name sst2 --tag 12 --parameter_fine_tuning True 
> 
> python main.py --mode output_hidden --dataset_name sst2 --tag 12 --parameter_fine_tuning False
> 
> 2.
> # 输出微调前后隐藏层的相似度，默认hidden_data_type = ['att_test', 'att_train', 'org_test','org_train'] 
> python main.py --mode output_sim --dataset_name sst2 
> ```

#### 5.对**攻击的训练样本**，做上面操作，再得到一个12维的相似度。 

> layers_sim,status
> "['0.9971', '0.9787', '0.9448', '0.8841', '0.8188', '0.789', '0.7278', '0.6424', '0.6429', '0.5878', '0.5045', '0.3942']",False

#### 6.训练一个分类器，输入是一个12维的相似度，输出结果是fake攻击还是truth干净

> layers_sim,status
> "['0.9935', '0.9703', '0.9516', '0.9205', '0.9102', '0.8916', '0.8792', '0.837', '0.7914', '0.6372', '0.4135', '0.288']",True
> "['0.9977', '0.9769', '0.9454', '0.8507', '0.7703', '0.734', '0.6551', '0.5683', '0.5649', '0.478', '0.5035', '0.5597']",False

```
#命令行训练语句，data_rate表示训练数据的比例【0，1】
python main.py --mode train_sim --dataset_name sst2 --data_rate 0.5
```

## 层精度检测方法：

1.用第一层微调后的bert（第12层训练的分类器），将每一层的隐藏层依次取出来（改model.py里的layer）

```
hidden_layer = 12 #修改这个层数,hidden_states[0]表示输入嵌入层（embedding layer）的输出，hidden_states[1]表示第1层隐藏层的输出
pooled_output = torch.mean(hidden_states[hidden_layer][:,1:,:],dim=1)     
pooled_output = self.dropout(pooled_output)
logits = self.classifier(pooled_output)
```

分别使用1到12层的hidden_layer预测每条数据的每个类别对应的置信度，每条数据每一层对应n个置信度（n是类别数），每条数据一共有n*12个置信度。classifier_12（hidden_states[i]），i从1到12，算出各个类别的概率：probabilities = torch.nn.functional.softmax(logits, dim=1)。

（1）输出第tag层的分类精度，tag从1到12。--pred_data_type att_test_succ/att_train_succ/train/test。添加df['status'] = True/False

```
python main.py --mode output_pred --dataset_name sst2 --max_seq_length 128 --batch_size 32 --tag 12
```

> 一共n*12个精度，c__{类别}__layer_{层},每个层的n个精度放一起，依次排列：
>
> c_0_layer_1,c_1_layer_1,c_0_layer_2,c_1_layer_2,c_0_layer_3,c_1_layer_3,c_0_layer_4,c_1_layer_4,c_0_layer_5,c_1_layer_5,c_0_layer_6,c_1_layer_6,c_0_layer_7,c_1_layer_7,c_0_layer_8,c_1_layer_8,c_0_layer_9,c_1_layer_9,c_0_layer_10,c_1_layer_10,c_0_layer_11,c_1_layer_11,c_0_layer_12,c_1_layer_12,status
> 0.4029,0.5971,0.4147,0.5853,0.3934,0.6066,0.327,0.6729,0.3944,0.6056,0.3471,0.6529,0.3587,0.6413,0.3355,0.6645,0.1373,0.8627,0.0166,0.9834,0.0007,0.9993,0.0,1.0,True
>
> 0.4598,0.5402,0.4962,0.5038,0.4534,0.5466,0.3875,0.6125,0.4867,0.5133,0.4261,0.5739,0.4271,0.5729,0.5048,0.4952,0.5385,0.4615,0.3954,0.6046,0.2601,0.7399,0.0087,0.9913,False

（2）将每一行的层分类精度转换为格式化的字符串列表，train (原始train)

```
python main.py --mode to_list --dataset_name sst2
```

> layers_pre,status
> 
> "['0.4029', '0.5971', '0.4147', '0.5853', '0.3934', '0.6066', '0.3270', '0.6730', '0.3944', '0.6056', '0.3471', '0.6529', '0.3587', '0.6413', '0.3355', '0.6645', '0.1373', '0.8627', '0.0166', '0.9834', '0.0007', '0.9993', '0.0000', '1.0000']",True
>
> "['0.4598', '0.5402', '0.4962', '0.5038', '0.4534', '0.5466', '0.3875', '0.6125', '0.4867', '0.5133', '0.4261', '0.5739', '0.4271', '0.5729', '0.5048', '0.4952', '0.5385', '0.4615', '0.3954', '0.6046', '0.2601', '0.7399', '0.0087', '0.9913']",False

（3）训练无监督分类器，训练集是原始的训练样本，测试集是原始个扰动的测试样本

```
python main.py --mode train_pred --dataset_name sst2
```

