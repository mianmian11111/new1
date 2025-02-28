

#### 1.微调的bert：只用一个分类器进行训练，用**第12层**的隐藏层向量作为这个分类器的输入，得到的模型有两个部分：一个是encoder_调，一个是mlp_调。	

> accuracy test: 0.9227
>
> ```
> python main.py --mode train --dataset_name sst2
> ```

#### 2.原始的bert：一个**原始的bert**，没有微调。模型只有一个部分：encoder_原。

> accuracy test: 0.4975

#### 3.攻击test和train数据，得到攻击成功的数据

```
1.textattack attack ...

# 处理攻击后的csv数据，得到att_test\att_train\org_test\org_train
2.python main.py --mode attack_csv_to_txt --dataset_name sst2
```

#### 4.对**干净的训练样本**（6920条），放入encoder_调 和 encoder_原， 对于每一层计算他们的隐藏层向量的相似度。 所以一个干净的样本将得到一个12维的相似度。

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

#### 5.对**攻击的训练样本**（5664条，隐藏层(13, 5664, 83, 768)），做上面操作，再得到一个12维的相似度。 

> accuracy test：微调前：0.3776，微调后：0.0477 （保存成csv）
>
> layers_sim,status
> "['0.9971', '0.9787', '0.9448', '0.8841', '0.8188', '0.789', '0.7278', '0.6424', '0.6429', '0.5878', '0.5045', '0.3942']",False

#### 6.训练一个mlp，输入是一个12维的相似度，输出结果是fake攻击还是truth干净

> layers_sim,status
> "['0.9935', '0.9703', '0.9516', '0.9205', '0.9102', '0.8916', '0.8792', '0.837', '0.7914', '0.6372', '0.4135', '0.288']",True
> "['0.9977', '0.9769', '0.9454', '0.8507', '0.7703', '0.734', '0.6551', '0.5683', '0.5649', '0.478', '0.5035', '0.5597']",False

```
#命令行训练语句，data_rate表示训练数据的比例【0，1】，默认1，layer_list表示检测用到的层，默认所有层
python main.py --mode train_sim --dataset_name sst2 --data_rate 0.5
```


