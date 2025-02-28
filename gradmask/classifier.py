# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import gc
import os
import math
import glob

from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import torch
import logging
import numpy as np
import torch.nn as nn
from overrides import overrides
from typing import List, Any, Dict, Union, Tuple
from tqdm import tqdm
from torchnlp.samplers.bucket_batch_sampler import BucketBatchSampler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizer
from utils.model import BertForSequenceClassification
from textattack.loggers.csv_logger import CSVLogger
from sklearn.metrics.pairwise import cosine_similarity


from args import ClassifierArgs
from utils.config import PRETRAINED_MODEL_TYPE, DATASET_TYPE
from data.reader import DataReader
from data.processor import DataProcessor
from data.instance import InputInstance
from data.dataset import ListDataset
from utils.metrics import Metric, RandomSmoothAccuracyMetrics, RandomAblationCertifyMetric
from utils.loss import ContrastiveLearningLoss, UnsupervisedCircleLoss
from utils.mask import mask_instance, mask_forbidden_index
from predictor import Predictor
from utils.utils import collate_fn, xlnet_collate_fn, convert_batch_to_bert_input_dict, build_forbidden_mask_words
from utils.hook import EmbeddingHook
from trainer import (BaseTrainer,
                    FreeLBTrainer,
                    PGDTrainer,
                    HotflipTrainer,
                    EmbeddingLevelMetricTrainer,
                    TokenLevelMetricTrainer,
                    RepresentationLearningTrainer,
                    MaskTrainer,
                    SAFERTrainer
                    )
from utils.textattack import build_english_attacker
from utils.textattack import CustomTextAttackDataset, SimplifidResult
from textattack.models.wrappers import HuggingFaceModelWrapper, HuggingFaceModelMaskEnsembleWrapper, HuggingFaceModelSaferEnsembleWrapper
from textattack.loggers.attack_log_manager import AttackLogManager
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
from utils.public import auto_create
from utils.certify import predict, lc_bound, population_radius_for_majority, population_radius_for_majority_by_estimating_lambda, population_lambda
from torch.optim.adamw import AdamW
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import ast
import joblib  # 用于保存和加载模型、
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from scipy.spatial.distance import cityblock
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



class Classifier:
    def __init__(self, args: ClassifierArgs):
        # check mode
        self.methods = {'train': self.train, 
                        'evaluate': self.evaluate,
                        'predict': self.predict, 
                        'attack': self.attack,
                        'augmentation': self.augmentation,
                        'certify': self.certify,
                        'statistics': self.statistics,
                        'output_hidden': self.output_hidden,
                        'train_sim': self.train_sim,
                        'evaluate_sim': self.evaluate_sim,
                        'output_pred': self.output_pred,
                        'train_pred': self.train_pred,
                        'evaluate_if_succ': self.evaluate_if_succ, 
                        'attack_csv_to_txt': self.attack_csv_to_txt,
                        'output_sim': self.output_sim,
                        'evaluate_pred': self.evaluate_pred,
                        'run_sim': self.run_sim,
                        'merge_hidden_states': self.merge_hidden_states,
                        'train_sim_supervised': self.train_sim_supervised,
                        'train_pred_supervised': self.train_pred_supervised
                        }# 'certify': self.certify}
        assert args.mode in self.methods, 'mode {} not found'.format(args.mode)

        # for data_reader and processing
        self.data_reader, self.tokenizer, self.data_processor = self.build_data_processor(args)
        self.model = self.build_model(args)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        if args.classifier == 'MLP':
            self.classifier = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100), max_iter=200, random_state=42)
        elif args.classifier == 'SVM':
            self.classifier = SVC(kernel='linear', random_state=42)  # 线性核
        elif args.classifier == 'DTC':
            self.classifie = DecisionTreeClassifier(random_state=42) # 决策树分类器
        elif args.classifier == 'LR':
            self.classifier = LogisticRegression(random_state=42)
        elif args.classifier == 'RFC':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)


        self.type_accept_instance_as_input = ['conat', 'sparse', 'safer']
        self.loss_function = self.build_criterion(args.dataset_name)
        
        self.forbidden_words = None
        if args.keep_sentiment_word:
            self.forbidden_words = build_forbidden_mask_words(args.sentiment_path)
    def save_model_to_file(self, save_dir: str, file_name: str):
        save_file_name = '{}.pth'.format(file_name)
        save_path = os.path.join(save_dir, save_file_name)
        torch.save(self.model.state_dict(), save_path)
        logging.info('Saving model to {}'.format(save_path))

    def loading_model_from_file(self, save_dir: str, file_name: str):
        load_file_name = '{}.pth'.format(file_name)
        load_path = os.path.join(save_dir, load_file_name)
        assert os.path.exists(load_path) and os.path.isfile(load_path), '{} not exits'.format(load_path)
        self.model.load_state_dict(torch.load(load_path), strict=False)
        logging.info('Loading model from {}'.format(load_path))
        return self.model

    def build_optimizer(self, args: ClassifierArgs, **kwargs):
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 遍历模型参数及其名称
        #for name, param in self.model.named_parameters():
            #print(f"Name: {name}, Size: {param.size()}")  # 打印每个参数的名称和尺寸
        
        #for n, p in self.model.named_parameters() :
            #if 'global_step' in n:
                #print("global_step在参数里面")
            #else:
                #print("global_step不在参数里面")
    
        # 定义参数分组
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and n != 'layer_weights'],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and n != 'layer_weights'],
                "weight_decay": 0.0,
            }, 
            {
                "params": [p for n, p in self.model.named_parameters() if 'layer_weights' in n],
                "weight_decay": 0.0,  # 通常不对layer_weights应用权重衰减
                "lr": args.learning_rate*7 # 使用特定的学习率
            } 

        ]
        
        # 创建优化器
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        return optimizer

    def build_model(self, args: ClassifierArgs) -> nn.Module:
        # config_class: PreTrainedConfig
        # model_class: PreTrainedModel
        config_class, model_class, _ = PRETRAINED_MODEL_TYPE.MODEL_CLASSES[args.model_type]
        # config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
        config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=self.data_reader.NUM_LABELS,
            finetuning_task=args.dataset_name,
            output_hidden_states=True,
        )
        # 用model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)来加载模型
        # 那么线性层 + 激活函数的权重就会随机初始化。
        # 我们的目的，就是通过微调，学习到线性层 + 激活函数的权重。
        model = model_class.from_pretrained(
            args.model_name_or_path,# bert-base-uncased 只包含 BertModel 的权重
            from_tf=bool('ckpt' in args.model_name_or_path),
            config=config
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]) 
        model.to(device)
        # 添加钩子以打印中间层输出
        # self.add_hooks(model)
        return model


    # def add_hooks(self, model):
    #     # 为BERT的每一层添加钩子
    #     for layer_num, layer in enumerate(model.encoder.layer):
    #         layer.register_forward_hook(
    #             lambda module, input, output, layer_num=layer_num: print(f"Layer {layer_num}: {output}")
    #         )
    


    def build_data_processor(self, args: ClassifierArgs, **kwargs) -> List[Union[DataReader, PreTrainedTokenizer, DataProcessor]]:
        data_reader = DATASET_TYPE.DATA_READER[args.dataset_name]()
        _, _, tokenizer_class = PRETRAINED_MODEL_TYPE.MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )
        data_processor = DataProcessor(data_reader=data_reader,
                                       tokenizer=tokenizer,
                                       model_type=args.model_type,
                                       max_seq_length=args.max_seq_length)

        return [data_reader, tokenizer, data_processor]

    def build_criterion(self, dataset):
        return DATASET_TYPE.get_loss_function(dataset)

    def build_data_loader(self, args: ClassifierArgs, data_type: str, tokenizer: bool = True, **kwargs) -> List[Union[Dataset, DataLoader]]:
        # for some training type, when training, the inputs type is Inputstance
        if data_type == 'train' and args.training_type in self.type_accept_instance_as_input:
            tokenizer = False
        shuffle = True if data_type == 'train' else False
        file_name = data_type
        if file_name == 'train' and args.file_name is not None:
            file_name = args.file_name
        dataset = auto_create('{}_max{}{}'.format(file_name, args.max_seq_length, '_tokenizer' if tokenizer else ''),
                            lambda: self.data_processor.read_from_file(args.dataset_dir, file_name, tokenizer=tokenizer),
                            True, args.caching_dir)
        
        # for collate function
        if tokenizer:
            collate_function = xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn
        else:
            collate_function = lambda x: x
        
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_function)
        return [dataset, data_loader]


    def build_attacker(self, args: ClassifierArgs, **kwargs):
        if args.training_type == 'sparse' or args.training_type == 'safer':
            if args.dataset_name in ['agnews', 'imdb']:
                batch_size = 300
            else:
                batch_size = 600
            if args.training_type == 'sparse':
                model_wrapper = HuggingFaceModelMaskEnsembleWrapper(args, 
                                            self.model, 
                                            self.tokenizer, 
                                            batch_size=batch_size)
            else:
                model_wrapper = HuggingFaceModelSaferEnsembleWrapper(args, 
                                                                    self.model, 
                                                                    self.tokenizer, 
                                                                    batch_size=batch_size)
        else:
            model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer, batch_size=args.batch_size)
        

        attacker = build_english_attacker(args, model_wrapper)
        return attacker

    def build_writer(self, args: ClassifierArgs, **kwargs) -> Union[SummaryWriter, None]:
        writer = None
        if args.tensorboard == 'yes':
            tensorboard_file_name = '{}-tensorboard'.format(args.build_logging_path())
            tensorboard_path = os.path.join(args.logging_dir, tensorboard_file_name)
            writer = SummaryWriter(tensorboard_path)
        return writer

    def build_trainer(self, args: ClassifierArgs, dataset: Dataset, data_loader: DataLoader) -> BaseTrainer:
        # get optimizer
        optimizer = self.build_optimizer(args)

        # get learning rate decay
        lr_scheduler = CosineAnnealingLR(optimizer, len(dataset) // args.batch_size * args.epochs)

        # get tensorboard writer
        writer = self.build_writer(args)

        trainer = BaseTrainer(data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        if args.training_type == 'freelb':
            trainer = FreeLBTrainer(data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'pgd':
            trainer = PGDTrainer(data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'advhotflip':
            trainer = HotflipTrainer(args, self.tokenizer, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'metric':
            trainer = EmbeddingLevelMetricTrainer(data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'metric_token':
            trainer = TokenLevelMetricTrainer(args, self.tokenizer, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'sparse':
            # trick = True if args.dataset_name in ['mr'] else False
            trainer = MaskTrainer(args, self.data_processor, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'safer':
            trainer = SAFERTrainer(args, self.data_processor, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        return trainer

    def train(self, args: ClassifierArgs):
        # get dataset
        dataset, data_loader = self.build_data_loader(args, 'train')

        # get trainer
        trainer = self.build_trainer(args, dataset, data_loader)

        best_metric = None
        for epoch_time in range(args.epochs):
            trainer.train_epoch(args, epoch_time)

            # saving model according to epoch_time
            self.saving_model_by_epoch(args, epoch_time)
            self.save_model_to_file(args.saving_dir, args.build_saving_file_name(description=f'best-{args.tag}'))

            # evaluate model according to epoch_time
            metric = self.evaluate(args, is_training=True)

            # update best metric
            # if best_metric is None, update it with epoch metric directly, otherwise compare it with epoch_metric
            if best_metric is None or metric > best_metric:
                best_metric = metric
                self.save_model_to_file(args.saving_dir, args.build_saving_file_name(description=f'best-{args.tag}'))
        
        if args.training_type == 'sparse' and args.incremental_trick and args.saving_last_epoch:
            self.save_model_to_file(args.saving_dir, args.build_saving_file_name(description=f'best-{args.tag}'))
        self.evaluate(args)
        
    @torch.no_grad()
    def evaluate(self, args: ClassifierArgs, is_training=False) -> Metric:
        if is_training:
            logging.info('Using current modeling parameter to evaluate')
            data_type = 'dev'
        else:
            # 是否加载微调后的参数，不加载就是微调前的bert模型
            if args.parameter_fine_tuning == True:
                self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description=f'best-{args.tag}')) # {args.tag}
            data_type = args.evaluation_data_type # test
        self.model.eval()

        dataset, data_loader = self.build_data_loader(args, data_type)
        epoch_iterator = tqdm(data_loader)

        metric = DATASET_TYPE.get_evaluation_metric(args.dataset_name,compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            assert isinstance(batch[0], torch.Tensor)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch = tuple(t.to(device) for t in batch)
            golds = batch[3]
            inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
            outputs = self.model.forward(**inputs)
            logits = outputs.logits# logits, (hidden_states), (attentions)这里输出的是隐藏层经过dropout和classifier的预测结果
            losses = self.loss_function(logits.view(-1, self.data_reader.NUM_LABELS), golds.view(-1))
            epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
            metric(losses, logits, golds)
        
        print(metric)
        logging.info(metric)
        return metric
    
    # 取出att_test攻击成功里能该表标签的数据
    @torch.no_grad()
    def evaluate_if_succ(self, args: ClassifierArgs, is_training=False):
        for attack in args.attack_list:
            if is_training:
                logging.info('Using current modeling parameter to evaluate')
                # data_type = 'dev'
            else:
                if args.parameter_fine_tuning:
                    self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description=f'best-{args.tag}'))
                # data_type = args.evaluation_data_type
            for data_type in args.attacked_data_type:
                org_data_dir = f'att_data/{attack}/org_{data_type}'
                self.model.eval()
                # 评估攻击前的数据
                dataset_org, data_loader_org = self.build_data_loader(args, org_data_dir)
                epoch_iterator_org = tqdm(data_loader_org)

                # 保存攻击前的预测结果
                predictions_org = []
                metric_org = DATASET_TYPE.get_evaluation_metric(args.dataset_name, compare_key=args.compare_key)

                for step, batch in enumerate(epoch_iterator_org):
                    assert isinstance(batch[0], torch.Tensor)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    batch = tuple(t.to(device) for t in batch)
                    golds = batch[3]
                    inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
                    
                    outputs = self.model.forward(**inputs)
                    logits = outputs.logits

                    # 获取攻击前的预测结果
                    predictions = torch.argmax(logits, dim=-1)
                    predictions_org.extend(predictions.cpu().numpy())
                    losses = self.loss_function(logits.view(-1, self.data_reader.NUM_LABELS), golds.view(-1))
                    metric_org(losses, logits, golds)

                # 评估攻击后的数据
                att_data_dir = f'att_data/{attack}/att_{data_type}'
                dataset_att, data_loader_att = self.build_data_loader(args, att_data_dir)
                epoch_iterator_att = tqdm(data_loader_att)
                
                # 保存攻击后的预测结果
                predictions_att = []
                metric_att = DATASET_TYPE.get_evaluation_metric(args.dataset_name, compare_key=args.compare_key)


                for step, batch in enumerate(epoch_iterator_att):
                    assert isinstance(batch[0], torch.Tensor)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    batch = tuple(t.to(device) for t in batch)
                    golds = batch[3]
                    inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
                    
                    outputs = self.model.forward(**inputs)
                    logits = outputs.logits

                    # 获取攻击后的预测结果
                    predictions = torch.argmax(logits, dim=-1)
                    predictions_att.extend(predictions.cpu().numpy())
                    losses = self.loss_function(logits.view(-1, self.data_reader.NUM_LABELS), golds.view(-1))
                    metric_att(losses, logits, golds)

                # 生成掩码，表示攻击前后预测结果是否不同,攻击前后预测结果不同则为True
                mask = (np.array(predictions_org) != np.array(predictions_att))

                # 将 mask 转换为 tensor
                filtered_mask = torch.tensor(mask)

                # 计算 True 的概率
                true_count = filtered_mask.sum().item()
                total_count = filtered_mask.size(0)

                # 计算 True 的概率
                true_probability = true_count / total_count if total_count > 0 else 0.0

                # 打印或记录 mask
                print("Mask (True indicates prediction changed):", filtered_mask)
                print(f"Probability of True in mask: {true_probability:.4f}")
                
                # 读取 test.txt 文件被攻击后的数据，evaluation_data_type是test
                with open(f'dataset/{args.dataset_name}/{att_data_dir}.txt', 'r') as att_file:
                    att_data = att_file.readlines()  # 读取所有行
                with open(f'dataset/{args.dataset_name}/{org_data_dir}.txt', 'r') as org_file:
                    org_data = org_file.readlines()

                # 确保 combined_mask 和数据集长度匹配
                assert len(filtered_mask) == len(att_data), "combined_mask length must match the length of the dataset."

                # 使用 combined_mask 筛选出攻击不成功的数据
                # combined_mask 为 True 表示攻击成功，反之为失败，保留combined_mask为True的数据
                filtered_att_data = [att_data[i] for i in range(len(filtered_mask)) if filtered_mask[i]]
                filtered_org_data = [org_data[i] for i in range(len(filtered_mask)) if filtered_mask[i]]

                att_data_succ = f'dataset/{args.dataset_name}/att_data/{attack}/att_{data_type}_succ.txt'
                org_data_succ = f'dataset/{args.dataset_name}/att_data/{attack}/org_{data_type}_succ.txt'
                # 将过滤后的扰动数据保存到新的文件
                with open(att_data_succ, 'w') as file:
                    file.writelines(filtered_att_data)
                print(f"att Filtered data saved to {att_data_succ}, total {len(att_data)} samples,save {len(filtered_att_data)}")

                # 将过滤后的原始数据保存到新的文件
                with open(org_data_succ, 'w') as file:
                    file.writelines(filtered_org_data)
                print(f"org Filtered data saved to {org_data_succ}, total {len(att_data)} samples,save {len(filtered_att_data)}")

                args.evaluation_data_type = f'att_data/{attack}/att_{data_type}_succ'
                metric_att_succ = self.evaluate(args)
                args.evaluation_data_type = f'att_data/{attack}/org_{data_type}_succ'
                metric_org_succ = self.evaluate(args)
                

                tag = args.evaluation_data_type
                print(f'{tag}_att:', metric_att)
                print(f'{tag}_org:', metric_org)
                print(f'succ_{tag}_att:', metric_att_succ)
                print(f'succ_{tag}_org:', metric_org_succ)
        return filtered_mask  # 返回 metric 和 mask
        
    @torch.no_grad()
    def output_pred(self, args: ClassifierArgs, is_training=False) -> Metric:
        if is_training:
            logging.info('Using current modeling parameter to evaluate')
            data_type = 'dev'
        else:
            # 是否加载微调后的参数，不加载就是微调前的bert模型
            if args.parameter_fine_tuning == True:
                self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description=f'best-12')) # {args.tag}
            data_type = args.pred_data_type # test
        self.model.eval()

        for attack in args.attack_list:
            for data_type in args.pred_data_type:
                data_dir = f'att_data/{attack}/{data_type}'
                dataset, data_loader = self.build_data_loader(args, data_dir)
                epoch_iterator = tqdm(data_loader)

                # 在循环外部初始化一个空列表来存储所有批次的正确类别概率
                all_correct_class_probabilities = []
                metric = DATASET_TYPE.get_evaluation_metric(args.dataset_name,compare_key=args.compare_key)
                for step, batch in enumerate(epoch_iterator):
                    assert isinstance(batch[0], torch.Tensor)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    batch = tuple(t.to(device) for t in batch)
                    golds = batch[3]
                    inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
                    outputs = self.model.forward(**inputs)
                    logits = outputs.logits# logits, (hidden_states), (attentions)这里输出的是隐藏层经过dropout和classifier的预测结果
                    # 算出各个类别的概率
                    probabilities = torch.nn.functional.softmax(logits, dim=1)  
                    # 分类结果（选择概率最高的类别）
                    # pred_labels = torch.argmax(probabilities, dim=1)    
                    # 获取每个样本正确类别的预测概率
                    correct_class_probabilities = probabilities[range(len(probabilities)), golds].tolist()
                    # 将当前批次的正确类别概率列表追加到总列表中
                    all_correct_class_probabilities.extend(correct_class_probabilities)
                    # print(f"\tpred_lable:{pred_labels},probabilities:{probabilities}\t")
                    print("len_correct_class_probabilities:", len(all_correct_class_probabilities))
                    losses = self.loss_function(logits.view(-1, self.data_reader.NUM_LABELS), golds.view(-1))
                    epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
                    metric(losses, logits, golds)
                
                # 将列表转换为 DataFrame
                df_new = pd.DataFrame(all_correct_class_probabilities, columns=['layer_{}'.format(args.tag)])
                # 保留四位小数
                df_new = df_new.round(4) 
                # 定义文件名
                file_name = f'dataset/{args.dataset_name}/sentence_layer_probility/{attack}/{data_type}.csv'
                if os.path.exists(file_name):
                # 读取现有的 CSV 文件
                    df_existing = pd.read_csv(file_name)
                    
                    # 1.将新数据追加到现有数据
                    #df_final = pd.concat([df_existing, df_new], axis=1)
                    # 保存到 CSV 文件
                    #df_final.to_csv(file_name, index=False)
                                    
                    # 2.添加标签列，status：True/False
                    if data_type == 'org_test' or data_type == 'org_train':
                        df_existing['status'] = True
                    else:
                        df_existing['status'] = False
                    df_existing.to_csv(file_name, index=False)
                    
                    print('Data saved to {}'.format(file_name))
                else:
                    # 如果文件不存在，直接使用新数据
                    df_new.to_csv(file_name, index=False)
                print(metric)
                logging.info(metric)
        return metric

    @torch.no_grad()
    def 层_evaluate(self, args: ClassifierArgs, is_training=False) -> Metric:
        if is_training:
            logging.info('Using current modeling parameter to evaluate')
            data_type = 'dev'
        else:
            # 是否加载微调后的参数，不加载就是微调前的bert模型
            if args.parameter_fine_tuning == True:
                self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
            data_type = args.evaluation_data_type # test
        self.model.eval()

        dataset, data_loader = self.build_data_loader(args, data_type)
        epoch_iterator = tqdm(data_loader)

        metric = DATASET_TYPE.get_evaluation_metric(args.dataset_name,compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            assert isinstance(batch[0], torch.Tensor)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch = tuple(t.to(device) for t in batch)
            golds = batch[3]
            inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
            outputs = self.model.forward(**inputs)
            logits_list = outputs.logits# logits(此处已改为12层logits的列表), (hidden_states), (attentions)这里输出的是隐藏层经过dropout和classifier的预测结果
            # print("logits_list_len:", len(logits_list))

            total_loss = 0 # 初始化总损失
            for logits in logits_list:
                # 分别算出每一层的loss，累加
                losses = self.loss_function(logits.view(-1, self.data_reader.NUM_LABELS), golds.view(-1))
                total_loss = total_loss + losses
                
            # 算出各个类别的概率
            # probabilities = torch.nn.functional.softmax(logits_list[-1], dim=1)  
            # 分类结果（选择概率最高的类别）
            # pred_labels = torch.argmax(probabilities, dim=1)    
            # print(f"\tpred_lable:{pred_labels},probabilities:{probabilities}\t") 
            epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(total_loss).item()))
            metric(total_loss, logits_list[args.tag], golds)

        print(metric)
        logging.info(metric)
        return metric


    # 处理每个批次时直接保存单个batch的隐藏层表示
    @torch.no_grad()
    def output_hidden_f(self, args: ClassifierArgs, is_training=False) -> Metric:
        tags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for i in tags:
            if args.parameter_fine_tuning:
                self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description=f'best-{i}'))
            self.model.eval()

            for attack in args.attack_list:
                for data_type in args.hidden_data_type:
                    print(f"data_type: {data_type + str(i)}")
                    data_dir = f'att_data/{attack}/{data_type}'

                    dataset, data_loader = self.build_data_loader(args, data_dir)
                    epoch_iterator = tqdm(data_loader)

                    metric = DATASET_TYPE.get_evaluation_metric(args.dataset_name, compare_key=args.compare_key)
                    for step, batch in enumerate(epoch_iterator):
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        batch = tuple(t.to(device) for t in batch)
                        golds = batch[3]
                        inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
                        outputs = self.model.forward(**inputs)
                        logits = outputs.logits
                        hidden_states = outputs.hidden_states

                        # 计算评估指标
                        probabilities = torch.nn.functional.softmax(logits, dim=1)
                        pred_labels = torch.argmax(probabilities, dim=1)
                        losses = self.loss_function(logits.view(-1, self.data_reader.NUM_LABELS), golds.view(-1))
                        epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
                        metric(losses, logits, golds)

                        # 直接保存当前批次的隐藏层表示
                        self.save_hidden_stage(args, hidden_states, data_dir, i, step)

                        gc.collect()
                        torch.cuda.empty_cache()

                    print(metric)
                    logging.info(metric)

        return metric


    def save_hidden_stage_f(self, args: ClassifierArgs, hidden_states, data_dir, layer, batch_step):
        """
        保存单个批次的隐藏层表示。
        :param args: ClassifierArgs 对象，包含配置信息。
        :param hidden_states: 当前批次的隐藏层表示，形状为 (13, batch_size, token_len, 768)。
        :param data_dir: 数据目录。
        :param layer: 当前处理的层。 
        :param batch_step: 当前批次的索引。
        """
        # 将张量转换为 NumPy 数组
        hidden_arrays = [tensor.cpu().numpy() for tensor in hidden_states]
        stacked_array = np.stack(hidden_arrays)  # 形状为 (13, batch_size, token_len, 768)

        # 提取指定层的隐藏层表示
        layer_array = stacked_array[layer]  # 形状为 (batch_size, token_len, 768)

        # 构造保存路径
        suffix = "fine-tuned" if args.parameter_fine_tuning else "original"
        save_dir = f'result_log/{args.dataset_name}_bert/hidden_states/{data_dir}'
        os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
        file_name = f'hidden_states_layer_{layer}_batch_{batch_step}_{suffix}.npy'
        save_path = os.path.join(save_dir, file_name)

        # 保存为 .npy 文件
        try:
            np.save(save_path, layer_array)
            print(f"Saved hidden states for batch {batch_step} to: {save_path}")
        except Exception as e:
            print(f"Failed to save hidden states to {save_path}. Error: {e}")
            

    def merge_hidden_states(self,args: ClassifierArgs):
        """
        合并所有批次的隐藏层文件，生成形状为 (13, n_sentences, token, 768) 的文件。
        :param args: ClassifierArgs 对象，包含配置信息。
        :param data_dir: 数据目录。
        """
        tuning = [True,False]
        for attack in args.attack_list:
            for data_type in args.hidden_data_type:
                for args.parameter_fine_tuning in tuning:
                    data_dir = f'att_data/{attack}/{data_type}'
                    suffix = "fine-tuned" if args.parameter_fine_tuning else "original"
                    save_dir = f'result_log/{args.dataset_name}_bert/hidden_states/{data_dir}'
            
                    # 初始化存储每一层的数组
                    merged_layers = []
                    
                    # 遍历每一层（0-12）
                    for layer in range(1,13):
                        # 按批次顺序找到所有该层的文件
                        file_pattern = os.path.join(save_dir, f'hidden_states_layer_{layer}_batch_*_{suffix}.npy')
                        file_paths = sorted(glob.glob(file_pattern))  # 按批次顺序排序
                        
                        # 初始化存储该层所有批次数据的列表
                        layer_data = []
                        
                        for file_path in file_paths:
                            # 加载批次文件
                            batch_data = np.load(file_path)  # 形状为 (batch_size, token, 768)
                            print(f"Layer {layer}, Batch data shape: {batch_data.shape}")
                            # 将批次数据添加到列表中
                            layer_data.append(batch_data)# 形状为 (num_batch, batch_size, 不同token, 768)
                        layer_data = self.combine_arrays_with_padding(layer_data)
                        for i in layer_data:
                            print(f"Layer {layer} batch reshape: {i.shape}")
                        
                        # 拼接所有批次的数据
                        if layer_data:
                            merged_layer = np.concatenate(layer_data, axis=0)  # 形状为 (n_sentences, token, 768)
                            print(f"Layer {layer}, Merged layer shape: {merged_layer.shape}")
                            merged_layers.append(merged_layer)
                        else:
                            print(f"No data found for layer {layer}. Skipping...")
                    
                    # 堆叠所有层的数据
                    if merged_layers:
                        merged_data = np.stack(merged_layers)  # 形状为 (12, n_sentences, token, 768)
                        print(f"Merged data shape: {merged_data.shape}")
                    else:
                        print("No data found. Merging failed.")
                        return
                    
                    if args.parameter_fine_tuning == True:
                        # 构造保存路径
                        merged_file_path = f'{save_dir}微调后.npy'
                    if args.parameter_fine_tuning == False:
                        # 构造保存路径
                        merged_file_path = f'{save_dir}原始.npy'
                        
                    # 保存合并后的数组(12, num_sentences, token, 768)
                    try:
                        np.save(merged_file_path, merged_data)
                        print(f"Merged hidden states saved to: {merged_file_path}")
                    except Exception as e:
                        print(f"Failed to save merged hidden states. Error: {e}")

    
    # 生成攻击前后数据的隐藏层，方便求相似度
    @torch.no_grad()
    def output_hidden(self, args: ClassifierArgs, is_training=False) -> Metric:
        # 是否加载微调后的参数，不加载就是微调前的bert模型
        if args.parameter_fine_tuning == True:
            self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description=f'best-12'))
        self.model.eval()
                

        for attack in args.attack_list:
            for data_type in args.hidden_data_type: 
                # print(f"data_type:", data_type)
                data_dir = f'att_data/{attack}/{data_type}'
                    

                dataset, data_loader = self.build_data_loader(args, data_dir)
                epoch_iterator = tqdm(data_loader)

                all_hidden_stages = []
                metric = DATASET_TYPE.get_evaluation_metric(args.dataset_name,compare_key=args.compare_key)
                for step, batch in enumerate(epoch_iterator):
                    assert isinstance(batch[0], torch.Tensor)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    batch = tuple(t.to(device) for t in batch)
                    golds = batch[3]
                    inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
                    outputs = self.model.forward(**inputs)
                    logits = outputs.logits# logits, (hidden_states), (attentions)这里输出的是隐藏层经过dropout和classifier的预测结果
                    hidden_states = outputs.hidden_states
                    all_hidden_stages.append(hidden_states)
                    # print("hidden_states[0].shape:",hidden_states[0].shape)#batch_size,token,768
                    # 算出各个类别的概率
                    probabilities = torch.nn.functional.softmax(logits, dim=1)  
                    # 分类结果（选择概率最高的类别）
                    pred_labels = torch.argmax(probabilities, dim=1)    
                    # print(f"\tpred_lable:{pred_labels},gold_lable:{golds},probabilities:{probabilities}\t") 
                    losses = self.loss_function(logits.view(-1, self.data_reader.NUM_LABELS), golds.view(-1))
                    epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
                    metric(losses, logits, golds)
                    gc.collect()
                    torch.cuda.empty_cache()


                #for i, stage in enumerate(all_hidden_stages):
                #torch.save(stage, f'result_log/sst2_bert/hidden_stages/hidden_stage_{i}.pt')
                print(metric)
                self.save_hidden_stage(args, all_hidden_stages, data_dir)
                logging.info(metric)
        return metric
    

    # 传入所有隐藏层的列表all_hidden_states，列表num_batch*元组(13个张量*batch_size,token,768)
    def save_hidden_stage(self, args: ClassifierArgs, all_hidden_states, data_dir):
        # 初始化一个空列表来存储所有数组
        all_hidden_array = []
        for hidden_stages in all_hidden_states:
            hidden_arrays = []
            # 遍历每个张量，转换为数组并添加到列表中
            for tensor in hidden_stages:
                # 将张量转换为 NumPy 数组
                array = tensor.cpu().numpy()
                hidden_arrays.append(array)
            # 使用 np.stack 堆叠所有数组
            # 注意：这假设所有张量的形状都是相同的                
            # 现在 stacked_array 的形状将是 (13, batch_size, token, 768)
            stacked_array = np.stack(hidden_arrays)
            all_hidden_array.append(stacked_array) # n*(13, batch_size, token, 768)
        # print("original:")
        # for i, arr in enumerate(all_hidden_array):
            # print(f"  Array {i}: {arr.shape}")    
        all_hidden_array = self.combine_arrays_with_padding(all_hidden_array)
        # print("Shapes of all_hidden_array after padding:")
        # for i, arr in enumerate(all_hidden_array):
            # print(f"  Array {i}: {arr.shape}")
        # 将所有(13, batch_size, token, 768)的数组沿着batch_size拼在一起
        stacked_arrays = np.concatenate(all_hidden_array, axis=1)
        print("shape of stacked_arrays:", stacked_arrays.shape)
        # 只取那一层的隐藏层输出layer=[0,12]，layer = 1代表取第1层，因为layer=0是输入bert的的嵌入层,num_batch*(batch_size, token, 768)
        # stacked_arrays = stacked_arrays[layer]
        # print("shape of layer_arrays:", stacked_arrays.shape)

        # 将堆叠后的数组存储为文件
        if args.parameter_fine_tuning == True:
            dir = f'result_log/{args.dataset_name}_bert/hidden_states/{data_dir}微调后.npy'
            np.save(dir, stacked_arrays)#改2：隐藏层存入哪里
        else:
            dir = f'result_log/{args.dataset_name}_bert/hidden_states/{data_dir}原始.npy'
            np.save(dir, stacked_arrays)#改2：隐藏层存入哪里
            
        print("success sava to:", dir)
        
    # 求'att_train', 'att_test', 'org_test', 'org_train'数据集在bert微调前后隐藏层的相似度
    def output_sim(self, args: ClassifierArgs):
        for attack in args.attack_list:
            if args.dataset_name == 'sst2':
                
                # 遍历每个 data 值
                for data_item in args.hidden_data_type:
                    # 构建文件路径
                    file_path1 = f'result_log/{args.dataset_name}_bert/hidden_states/att_data/{attack}/{data_item}原始.npy'
                    file_path2 = f'result_log/{args.dataset_name}_bert/hidden_states/att_data/{attack}/{data_item}微调后.npy'
                        
                    # 加载 .npy 文件,(12, num_sentences, token, 768)
                    array1 = np.load(file_path1)
                    array2 = np.load(file_path2)
                        
                    print("array1.shape:", array1.shape)
                    print("array2.shape:", array2.shape)
                        
                    # 获取句子的数量
                    n_sentences = array1.shape[1]  # 假设第二维是句子数量
                        
                    # 计算每个句子的每层平均表示，忽略第一个 token
                    avg_array1 = np.mean(array1[:, :, 1:, :], axis=2)  # 形状 (13, n_sentences, 768)
                    avg_array2 = np.mean(array2[:, :, 1:, :], axis=2)  # 形状 (13, n_sentences, 768)
                       
                    for sim in args.sim_f:
                        # 存储相似度
                        similarity_matrix = np.zeros((n_sentences, 12))  # (n_sentences, 12)
                        # 遍历每个句子，计算相似度
                        for layer in range(1, 13):
                            # 提取当前层的平均表示
                            layer_array1 = avg_array1[layer]  # 形状为 (n_sentences, 768)
                            layer_array2 = avg_array2[layer]  # 形状为 (n_sentences, 768)
                            
                            if sim == 'cosine_similarity':
                                # 计算余弦相似度矩阵
                                sim_matrix = cosine_similarity(layer_array1, layer_array2)
                                # 提取对角线元素（即每个句子与其对应的句子的相似度）
                                similarity_matrix[:, layer] = np.diag(sim_matrix)
                            if sim == '点积':
                                # 计算点积
                                dot_product = np.sum(layer_array1 * layer_array2, axis=1)  # 按行计算点积，形状为 (n_sentences,)
                                # 将点积结果存储到相似度矩阵中
                                similarity_matrix[:, layer] = dot_product
                            if sim == '欧式距离':
                                euclidean_distance = np.linalg.norm(layer_array1 - layer_array2, axis=1)  # 形状为 (n_sentences,)
                                similarity_matrix[:, layer] = euclidean_distance
                            if sim =='皮尔逊相关系数':
                                # 计算皮尔逊相关系数
                                pearson_corr = np.zeros(layer_array1.shape[0])  # 初始化存储相关系数的数组，形状为 (n_sentences,)
                                for i in range(layer_array1.shape[0]):
                                    # 计算每对向量之间的皮尔逊相关系数
                                    corr = np.corrcoef(layer_array1[i], layer_array2[i])[0, 1]
                                    pearson_corr[i] = corr
                                similarity_matrix[:, layer] = pearson_corr
                            if sim == '曼哈顿距离':
                                manhattan_distance = np.sum(np.abs(layer_array1 - layer_array2), axis=1)  # 形状为 (n_sentences,)
                                similarity_matrix[:, layer] = manhattan_distance  # 存储曼哈顿距离
                                                                                
                        # 将每一层相似度转为字符串列表并保留四位小数
                        layers_sim = similarity_matrix.round(4).astype(str).tolist()  # 保留四位小数并转换为字符串列表
                            
                        # 创建一个 DataFrame，并将 layers_sim 作为一列
                        df = pd.DataFrame({'layers_sim': layers_sim})
                            
                        # 将 status 列添加到 DataFrame
                        if data_item  == 'att_train' or data_item  == 'att_test':
                            df['status'] = False# 将状态列的值设为 True
                        if data_item  == 'org_train' or data_item  == 'org_test':
                            df['status'] = True# 将状态列的值设为 True
                        # 构建输出文件路径
                            
                        output_file = f'dataset/{args.dataset_name}/sentence_layer_sim/bert_if_fine_tuning/{attack}/相似度/{sim}/{data_item}.csv'
                            
                        # 将结果保存到 CSV 文件，只包含 layers_sim 和 status 列
                        df.to_csv(output_file, columns=['layers_sim', 'status'], index=False)                
                        print(f"Results for {data_item} saved to {output_file}, num of sentences:", n_sentences)
            # 完善
            if args.dataset_name == 'imdb' or 'agnews':
                # 定义 tag 值列表和数据列表
                tags = [1,2,3,4]
                # 遍历每个 data 值
                for data_item in args.hidden_data_type:
                    for i in tags:
                        # 构建文件路径
                        file_path1 = f'result_log/{args.dataset_name}_bert/hidden_states/att_data/{attack}/{data_item}{i}12原始.npy'
                        file_path2 = f'result_log/{args.dataset_name}_bert/hidden_states/att_data/{attack}/{data_item}{i}12微调后.npy'
                            
                        # 加载 .npy 文件
                        array1 = np.load(file_path1)
                        array2 = np.load(file_path2)
                            
                        print("array1.shape:", array1.shape)
                        print("array2.shape:", array2.shape)
                            
                        # 获取句子的数量
                        n_sentences = array1.shape[1]  # 假设第二维是句子数量
                            
                        # 计算每个句子的每层平均表示，忽略第一个 token
                        avg_array1 = np.mean(array1[:, :, 1:, :], axis=2)  # 形状 (13, n_sentences, 768)
                        avg_array2 = np.mean(array2[:, :, 1:, :], axis=2)  # 形状 (13, n_sentences, 768)
                            
                        # 存储相似度
                        similarity_matrix = np.zeros((n_sentences, 12))  # (n_sentences, 12)
                            
                        # 遍历每个句子，计算相似度
                        for sentence in range(n_sentences):
                            for layer in range(12):
                                similarity_matrix[sentence, layer] = cosine_similarity(
                                    avg_array1[layer:layer+1, sentence].reshape(1, -1),  # 第 layer 层的平均表示
                                    avg_array2[layer:layer+1, sentence].reshape(1, -1)   # 第 layer 层的平均表示
                                )[0, 0]
                            
                        # 将每一层相似度转为字符串列表并保留四位小数
                        layers_sim = similarity_matrix.round(4).astype(str).tolist()  # 保留四位小数并转换为字符串列表
                            
                        # 创建一个 DataFrame，并将 layers_sim 作为一列
                        df = pd.DataFrame({'layers_sim': layers_sim})
                            
                        # 将 status 列添加到 DataFrame
                        if data_item  == 'att_train' or data_item  == 'att_test':
                            df['status'] = False# 将状态列的值设为 True
                        if data_item  == 'org_train' or data_item  == 'org_test':
                            df['status'] = True# 将状态列的值设为 True
                        # 构建输出文件路径
                            
                        output_file = f'dataset/{args.dataset_name}/sentence_layer_sim/bert_if_fine_tuning/{attack}/{data_item}{i}.csv'
                            
                        # 将结果保存到 CSV 文件，只包含 layers_sim 和 status 列
                        df.to_csv(output_file, columns=['layers_sim', 'status'], index=False)
                            
                        print(f"Results for {data_item}{i} saved to {output_file}, num of sentences:", n_sentences)


    # 填充all_hidden_array
    def combine_arrays_with_padding(self, all_hidden_array):
        # 找出最大的 token 长度
        max_tokens = max(arr.shape[2] for arr in all_hidden_array)
        # max_tokens = max(arr.shape[1] for arr in all_hidden_array)
    
        padded_arrays = []
        for arr in all_hidden_array:
            # 计算需要填充的数量
            pad_size = max_tokens - arr.shape[2]
            # pad_size = max_tokens - arr.shape[1]
            # 创建填充
            pad_width = ((0, 0), (0, 0), (0, pad_size), (0, 0))
            # pad_width = ((0, 0), (0, pad_size), (0, 0))
            # 应用填充
            padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
            padded_arrays.append(padded_arr)
    
        return padded_arrays
    
    def train_sim(self, args: ClassifierArgs):
        for attack in args.attack_list:
            # 读取 CSV 文件，读取原始训练集的相似度,数据量只取训练数据的args.data_rate,默认100%
            df = pd.read_csv(f'dataset/{args.dataset_name}/sentence_layer_sim/bert_if_fine_tuning/{attack}/org_train.csv').sample(frac=args.data_rate)

            # 将 layers_sim 列中的字符串转换为列表，并只保留layer_list里对应层的元素，默认所有层
            df['layers_sim'] = df['layers_sim'].apply(lambda x: [ast.literal_eval(x)[i] for i in args.layer_list])

            # 将列表转换为二维数组
            X = df['layers_sim'].tolist()

            # 将列表中的字符串转换为浮点数
            X = [[float(num) for num in sublist] for sublist in X]

            # 读取测试集的相似度，测试集包含原始和被攻击后的测试样本的相似度
            mixed_df = pd.read_csv('dataset/{args.dataset_name}/sentence_layer_sim/bert_if_fine_tuning/{attack}/处理后的/all_test.csv')

            # 将 layers_sim 列中的字符串转换为列表，读取对应层
            mixed_df['layers_sim'] = mixed_df['layers_sim'].apply(lambda x: [ast.literal_eval(x)[i] for i in args.layer_list])

            # 将列表转换为二维数组
            X_mixed = mixed_df['layers_sim'].tolist()

            # 将列表中的字符串转换为浮点数
            X_mixed = [[float(num) for num in sublist] for sublist in X_mixed]

            # 将 status 列中的布尔值转换为整数，True 表示正常样本 (1)，False 表示异常样本 (-1)
            y_true = mixed_df['status'].apply(lambda x: 1 if x else -1).values

            # 创建 OneClassSVM 模型实例并训练
            clf = OneClassSVM(nu=0.06, kernel='rbf', gamma='scale').fit(X)

            # 预测
            predictions = clf.predict(X_mixed)
            #print("预测结果:", predictions)

            # 计算异常分数
            scores = clf.score_samples(X_mixed)
            #print("异常分数:", scores)

            # 计算准确率
            accuracy = accuracy_score(y_true, predictions)
            print("准确率:", accuracy)

            # 输出分类报告
            # 输出分类报告，保留4位小数
            print("分类报告:\n", classification_report(y_true, predictions, target_names=['Anomaly', 'Normal'], digits=4))

            # 计算ROC曲线的坐标点
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)

            # 计算AUC值
            auc = roc_auc_score(y_true, scores)
            print("AUC值:", auc)

            # 显示图像
            plt.show()
            
    def train_pred(self, args: ClassifierArgs):
        for attack in args.attack_list:
            # 读取 CSV 文件，读取原始训练集的相似度,数据量只取训练数据的args.data_rate,默认100%
            df = pd.read_csv(f'dataset/{args.dataset_name}/sentence_layer_probility/{attack}/org_train.cs').sample(frac=args.data_rate)

            # 将 layers_sim 列中的字符串转换为列表，并只保留layer_list里对应层的元素，默认所有层
            df['layers_pre'] = df['layers_pre'].apply(lambda x: [ast.literal_eval(x)[i] for i in args.layer_list])

            # 将列表转换为二维数组
            X = df['layers_pre'].tolist()

            # 将列表中的字符串转换为浮点数
            X = [[float(num) for num in sublist] for sublist in X]

            # 读取测试集的相似度，测试集包含原始和被攻击后的测试样本的相似度
            mixed_df = pd.read_csv('dataset/{args.dataset_name}/sentence_layer_probility/{attack}/all_test.csv')

            # 将 layers_sim 列中的字符串转换为列表，读取对应层
            mixed_df['layers_pre'] = mixed_df['layers_pre'].apply(lambda x: [ast.literal_eval(x)[i] for i in args.layer_list])

            # 将列表转换为二维数组
            X_mixed = mixed_df['layers_pre'].tolist()

            # 将列表中的字符串转换为浮点数
            X_mixed = [[float(num) for num in sublist] for sublist in X_mixed]

            # 将 status 列中的布尔值转换为整数，True 表示正常样本 (1)，False 表示异常样本 (-1)
            y_true = mixed_df['status'].apply(lambda x: 1 if x else -1).values

            # 创建 OneClassSVM 模型实例并训练
            clf = OneClassSVM(nu=0.06, kernel='rbf', gamma='scale').fit(X)

            # 预测
            predictions = clf.predict(X_mixed)
            #print("预测结果:", predictions)

            # 计算异常分数
            scores = clf.score_samples(X_mixed)
            #print("异常分数:", scores)

            # 计算准确率
            accuracy = accuracy_score(y_true, predictions)
            print("准确率:", accuracy)

            # 输出分类报告
            # 输出分类报告，保留4位小数
            print("分类报告:\n", classification_report(y_true, predictions, target_names=['Anomaly', 'Normal'], digits=4))

            # 计算ROC曲线的坐标点
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)

            # 计算AUC值
            auc = roc_auc_score(y_true, scores)
            print("AUC值:", auc)

            # 显示图像
            plt.show()
            
    def train_pred_supervised(self, args: ClassifierArgs):
        for attack in args.attack_list:
            # 读取训练集和测试集
            train_df = pd.read_csv(f'dataset/{args.dataset_name}/sentence_layer_probility/{attack}/all_train.csv')  # 替换为你的训练集文件路径
            test_df = pd.read_csv(f'dataset/{args.dataset_name}/sentence_layer_probility/{attack}/all_test.csv')  # 替换为你的测试集文件路径

            # 将 layers_pre 列中的字符串转换为列表
            train_df['layers_pre'] = train_df['layers_pre'].apply(ast.literal_eval)
            test_df['layers_pre'] = test_df['layers_pre'].apply(ast.literal_eval)

            # 提取特征和标签
            X_train = np.array(train_df['layers_pre'].tolist())  # 训练特征
            y_train = train_df['status'].values  # 训练标签

            X_test = np.array(test_df['layers_pre'].tolist())  # 测试特征
            y_test = test_df['status'].values  # 测试标签

            # 训练模型
            self.classifier.fit(X_train, y_train)

            # 预测
            y_pred = self.classifier.predict(X_test)
            
            # 保存模型
            joblib.dump(self.classifier, f'save_models/{args.dataset_name}_bert/pred_models/{attack}/{args.classifier}.pkl')
            print(f"Model trained and saved as {args.classifier}.pkl")
            
            self.evaluate_pred(args, attack, is_training=True)
        
        
    def evaluate_pred(self, args: ClassifierArgs, attack, is_training = False):
        if is_training:
            print('Using current modeling parameter to evaluate')
        else:
            # 加载模型
            self.classifier = joblib.load(f'save_models/{args.dataset_name}_bert/pred_models/{attack}/{args.classifier}.pkl')
            print("load model from file.")
            
        # 读取 CSV 文件
        df = pd.read_csv(f'dataset/{args.dataset_name}/sentence_layer_probility/{attack}/all_test.csv')

        # 将 layers_sim 列中的字符串转换为列表
        df['layers_pre'] = df['layers_pre'].apply(ast.literal_eval)

        # 提取特征和标签
        X = np.array(df['layers_pre'].tolist())  # 特征：layers_sim 列
        y = df['status'].values  # 标签：status 列

        # 预测
        y_pred = self.classifier.predict(X)  # 直接使用整个特征集进行预测

        # 评估模型
        print("Accuracy:", accuracy_score(y, y_pred))
        print(classification_report(y, y_pred))

    

    def train_sim_supervised(self, args: ClassifierArgs):
        for attack in args.attack_list:
            print(f"attack: {attack}")
            # 读取 CSV 文件
            df = pd.read_csv(f'dataset/{args.dataset_name}/sentence_layer_sim/bert_if_fine_tuning/{attack}/all_train.csv')

            # 将 layers_sim 列中的字符串转换为列表
            df['layers_sim'] = df['layers_sim'].apply(ast.literal_eval)

            # 提取特征和标签,前三个
            # X = np.array([np.array(layer[-4:]).astype(float) for layer in df['layers_sim']])
            X = np.array(df['layers_sim'].tolist())  # 特征：layers_sim 列
            y = df['status'].values  # 标签：status 列

            # 训练模型，使用整个train数据集
            self.classifier.fit(X, y)
            
            # 保存模型
            joblib.dump(self.classifier, f'save_models/{args.dataset_name}_bert/sim_models/{attack}/{args.classifier}.pkl')
            print(f"Model trained and saved as {args.classifier}.pkl")
            
            self.evaluate_sim(args, attack, is_training=True)



    def evaluate_sim(self, args: ClassifierArgs, is_training = False):
        if is_training:
            print('Using current modeling parameter to evaluate')
        else:
            # 加载模型
            self.classifier = joblib.load(f'save_models/{args.dataset_name}_bert/sim_models/{args.attack}/{args.classifier}.pkl')
            print("load model from file.")
            
        # 读取 CSV 文件
        df = pd.read_csv(f'dataset/{args.dataset_name}/sentence_layer_sim/bert_if_fine_tuning/{args.attack}/处理后的/all_test.csv')

        # 假设 df 是你的 DataFrame
        
        nan_rows = df[df['layers_sim'].isna()]
        # 输出这些行
        print(nan_rows)
        # 检查 'layers_sim' 列中的值是否为 nan，如果是，则替换为一个有效的字面量结构，比如空列表 []
        # df['layers_sim'] = df['layers_sim'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
        # 将 layers_sim 列中的字符串转换为列表
        df['layers_sim'] = df['layers_sim'].apply(ast.literal_eval)

        # 提取特征和标签
        # X = np.array([np.array(layer[-4:]).astype(float) for layer in df['layers_sim']])

        X = np.array(df['layers_sim'].tolist())  # 特征：layers_sim 列
        y = df['status'].values  # 标签：status 列

        # 预测
        y_pred = self.classifier.predict(X)  # 直接使用整个特征集进行预测
        y_pred_proba = self.classifier.predict_proba(X)[:, 1]  # 预测概率（假设二分类问题）
        
        
        # 评估模型
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy: {accuracy:.4f}")  # 输出精度为4位小数的准确率

        # 输出分类报告
        print("Classification Report:")
        print(classification_report(y, y_pred, digits=4))  # digits 参数设置输出精度为4位小数

        # 计算 AUC 值
        auc = roc_auc_score(y, y_pred_proba)
        print(f"AUC: {auc:.4f}")  # 输出精度为4位小数的 AUC 值
        
    # 将攻击后的train和test数据的attack_results.csv文件，分别取出att_test.txt、org_test.txt、att_train.txt、org_train.txt
    def attack_csv_to_txt(self, args: ClassifierArgs):
        # data_type依次取['train', 'test']
        for attack in args.attack_list:
            for data_type in args.attacked_data_type:
                # 读取 CSV 文件,一个test.csv文件生成两个txt文件，分别保存perturbed_text/original_text和 ground_truth_output
                csv_file = f'result_log/{args.dataset_name}_bert/attack-len256-epo10-batch32/{attack}/{data_type}/attack_results.csv'  # 替换为你的 CSV 文件路径
                df = pd.read_csv(csv_file)

                # 筛选出 result_type 为 "Successful" 的行
                successful_attacks = df[df['result_type'] == 'Successful']

                # 分别取出perturbed_text和original_text
                text_names = ['original_text', 'perturbed_text']

                for text_name in text_names:
                    # 提取 perturbed_text/original_text和 ground_truth_output 列 
                    get_texts = successful_attacks[text_name].astype(str)
                    ground_truth_outputs = successful_attacks['ground_truth_output'].astype(int)

                    if text_name == 'original_text':
                        # 将结果保存到 TXT 文件中
                        txt_file = f'dataset/{args.dataset_name}/att_data/{attack}/org_{data_type}.txt'  # 输出的 TXT 文件路径

                        with open(txt_file, 'w') as f:
                            for text, output in zip(get_texts, ground_truth_outputs):
                                # 删除 perturbed_text 中的 '[[', ']]'
                                text = text.replace('[[', '').replace(']]', '')
                                f.write(f"{text}\t{output}\n")
                            print(f"Successfully saved {data_type} {text_name} and ground_truth_output to {txt_file}")
                    else:
                        txt_file = f'dataset/{args.dataset_name}/att_data/{attack}/att_{data_type}.txt'  # 输出的 TXT 文件路径

                        with open(txt_file, 'w') as f:
                            for text, output in zip(get_texts, ground_truth_outputs):
                                # 删除 perturbed_text 中的 '[[', ']]'
                                text = text.replace('[[', '').replace(']]', '')
                                f.write(f"{text}\t{output}\n")
                            print(f"Successfully saved {data_type} {text_name} and ground_truth_output to {txt_file}")
            args.evaluation_data_type = 'train'
            self.evaluate_if_succ(args)
            args.evaluation_data_type = 'test'
            self.evaluate_if_succ(args)
        
    @torch.no_grad()
    def infer(self, args: ClassifierArgs) -> Dict:
        content = args.content
        assert content is not None, 'in infer mode, parameter content cannot be None! '
        content = content.strip()
        assert content != '' and len(content) != 0, 'in infer mode, parameter content cannot be empty! '

        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description=f'best-{args.tag}'))
        self.model.eval()

        predictor = Predictor(self.model, self.data_processor, args.model_type)
        pred_probs = predictor.predict(content)
        pred_label = np.argmax(pred_probs)
        pred_label = self.data_reader.get_idx_to_label(pred_label)
        if pred_label == '100':
            pred_label = '0'
        elif pred_label == '101':
            pred_label = '1'

        result_in_dict = {'content': content, 'pred_label':pred_label, 'pred_confidence': pred_probs}
        result_in_str = ', '.join(['{}: {}'.format(key, value)
                                   if not isinstance(value, list)
                                   else '{}: [{}]'.format(key, ', '.join(["%.4f" % val for val in value]))
                                   for key, value in result_in_dict.items()])
        print(result_in_str)
        logging.info(result_in_str)
        return result_in_dict

    # for sparse adversarial training with random mask,
    # predict() is to get the smoothing result, which is different from evaluate()

    @torch.no_grad()
    def predict(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description=f'best-{args.tag}'))
        self.model.eval()
        predictor = Predictor(self.model, self.data_processor, args.model_type)

        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, tokenizer=False)
        assert isinstance(dataset, ListDataset)
        if args.predict_numbers == -1:
            predict_dataset = dataset.data
        else:
            predict_dataset = np.random.choice(dataset.data, size=(args.predict_numbers, ), replace=False)
        
        description = tqdm(predict_dataset)
        metric = RandomSmoothAccuracyMetrics()
        for data in description:
            tmp_instances = self.mask_instance_decorator(args, data, args.predict_ensemble)
            tmp_probs = predictor.predict_batch(tmp_instances)
            target = self.data_reader.get_label_to_idx(data.label)
            pred = predict(tmp_probs, args.alpha)
            metric(pred, target)
            description.set_description(metric.__str__())
        print(metric)
        logging.info(metric)
        
    
    def attack(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description=f'best-{args.tag}'))
        self.model.eval()

        # build test dataset 
        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, tokenizer=False)
        test_instances = dataset.data
       
        # build attacker
        attacker = self.build_attacker(args)

        attacker_log_path = '{}'.format(args.build_logging_path())
        attacker_log_path = os.path.join(args.logging_dir, attacker_log_path)
        attacker_log_manager = AttackLogManager()
        # attacker_log_manager.enable_stdout()
        attacker_log_manager.add_output_file(os.path.join(attacker_log_path, '{}.txt'.format(args.attack_method)))
        
        for i in range(args.attack_times):
            print("Attack time {}".format(i))
            
            choice_instances = np.random.choice(test_instances, size=(args.attack_numbers,),replace=False)
            dataset = CustomTextAttackDataset.from_instances(args.dataset_name, choice_instances, self.data_reader.get_labels())
            results_iterable = attacker.attack_dataset(dataset)
            description = tqdm(results_iterable, total=len(choice_instances))
            result_statistics = SimplifidResult()
            for result in description:
                try:
                    attacker_log_manager.log_result(result)
                    result_statistics(result)
                    description.set_description(result_statistics.__str__())
                except RuntimeError as e:
                    print('error in process')

        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def 改_attack(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()

        # build test dataset 
        dataset, _ = self.build_data_loader(args, args.attack_data_type, tokenizer=False)
        test_instances = dataset.data
       
        # build attacker
        attacker = self.build_attacker(args)

        attacker_log_path = '{}'.format(args.build_logging_path())
        attacker_log_path = os.path.join(args.logging_dir, attacker_log_path)
        attacker_log_manager = AttackLogManager()
        # attacker_log_manager.enable_stdout()
        attacker_log_manager.add_output_file(os.path.join(attacker_log_path, '{}.txt'.format(args.attack_method)))
        # 实例化CSVLogger
        csv_logger = CSVLogger(filename=os.path.join(attacker_log_path, 'attack_results.csv'), color_method='file')
    
        for i in range(args.attack_times):
            print("Attack time {}".format(i))
            
           # choice_instances = np.choice(test_instances, size=(args.attack_numbers,),replace=False)
            choice_instances = np.random.choice(test_instances, size=(args.attack_numbers,),replace=False)
            # choice_instances = test_instances[:args.attack_numbers]
            dataset = CustomTextAttackDataset.from_instances(args.dataset_name, choice_instances, self.data_reader.get_labels())
            results_iterable = attacker.attack_dataset(dataset)
            description = tqdm(results_iterable, total=len(choice_instances))
            result_statistics = SimplifidResult()
            for result in description:
                try:
                    attacker_log_manager.log_result(result)
                    result_statistics(result)
                    description.set_description(result_statistics.__str__())
                    
                     # 记录每条攻击结果到CSV文件
                    csv_logger.log_attack_result(result)
                except RuntimeError as e:
                    print('error in process')

        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()
        # 将CSV数据写入文件
        csv_logger.flush()

    def augmentation(self, args: ClassifierArgs, **kwargs):
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description=f'best-{args.tag}'))
        self.model.eval()

        train_instances, _ = self.build_data_loader(args, 'train', tokenizer=False)
        train_dataset_len = len(train_instances.data)
        print('Training Set: {} sentences. '.format(train_dataset_len))
        
        # delete instance whose length is smaller than 3
        train_instances_deleted = [instance for instance in train_instances.data if instance.length() >= 3]
        dataset_to_aug = np.random.choice(train_instances_deleted, size=(int(train_dataset_len * 0.5), ), replace=False)

        dataset_to_write = np.random.choice(train_instances.data, size=(int(train_dataset_len * 0.5), ), replace=False).tolist()
        attacker = self.build_attacker(args)
        attacker_log_manager = AttackLogManager()
        dataset = CustomTextAttackDataset.from_instances(args.dataset_name, dataset_to_aug, self.data_reader.get_labels())
        results_iterable = attacker.attack_dataset(dataset)
        aug_instances = []
        for result, instance in tqdm(zip(results_iterable, dataset_to_aug), total=len(dataset)):
            try:
                adv_sentence = result.perturbed_text()
                aug_instances.append(InputInstance.from_instance_and_perturb_sentence(instance, adv_sentence))
            except:
                print('one error happend, delete one instance')

        dataset_to_write.extend(aug_instances)
        self.data_reader.saving_instances(dataset_to_write, args.dataset_dir, 'aug_{}'.format(args.attack_method))
        print('Writing {} Sentence. '.format(len(dataset_to_write)))
        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()


    def certify(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()
        predictor = Predictor(self.model, self.data_processor, args.model_type)

        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, tokenizer=False)
        assert isinstance(dataset, ListDataset)
        if args.certify_numbers == -1:
            certify_dataset = dataset.data
        else:
            certify_dataset = np.random.choice(dataset.data, size=(args.certify_numbers, ), replace=False)
        
        description = tqdm(certify_dataset)
        num_labels = self.data_reader.NUM_LABELS
        metric = RandomAblationCertifyMetric() 
        for data in description:
            target = self.data_reader.get_label_to_idx(data.label)
            data_length = data.length()
            keep_nums = data_length - round(data_length * args.sparse_mask_rate)

            tmp_instances = self.mask_instance_decorator(args, data, args.predict_ensemble)
            tmp_probs = predictor.predict_batch(tmp_instances)
            guess = np.argmax(np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels))

            if guess != target:
                metric(np.nan, data_length)
                continue
                
            tmp_instances = self.mask_instance_decorator(args, data, args.ceritfy_ensemble)
            tmp_probs = predictor.predict_batch(tmp_instances)
            guess_counts = np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels)[guess]
            lower_bound, upper_bound = lc_bound(guess_counts, args.ceritfy_ensemble, args.alpha)
            if args.certify_lambda:
                # tmp_instances, mask_indexes = mask_instance(data, args.sparse_mask_rate, self.tokenizer.mask_token,nums=args.ceritfy_ensemble * 2, return_indexes=True)
                # tmp_probs = predictor.predict_batch(tmp_instances)
                # tmp_preds = np.argmax(tmp_probs, axis=-1)
                # ablation_indexes = [list(set(list(range(data_length))) - set(indexes.tolist())) for indexes in mask_indexes]
                # radius = population_radius_for_majority_by_estimating_lambda(lower_bound, data_length, keep_nums, tmp_preds, ablation_indexes, num_labels, guess, samplers = 200)
                radius = population_radius_for_majority(lower_bound, data_length, keep_nums, lambda_value=guess_counts / args.ceritfy_ensemble)
            else:
                radius = population_radius_for_majority(lower_bound, data_length, keep_nums)
            
            metric(radius, data_length)

            result = metric.get_metric()
            description.set_description("Accu: {:.2f}%, Median: {}".format(result['accuracy'] * 100, result['median']))
        print(metric)
        logging.info(metric)

        # logging metric certify_radius and length
        logging.info(metric.certify_radius())
        logging.info(metric.sentence_length())


    def statistics(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()
        predictor = Predictor(self.model, self.data_processor, args.model_type)

        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, tokenizer=False)
        assert isinstance(dataset, ListDataset)
        if args.certify_numbers == -1:
            certify_dataset = dataset.data
        else:
            certify_dataset = np.random.choice(dataset.data, size=(args.certify_numbers, ), replace=False)
        
        description = tqdm(certify_dataset)
        num_labels = self.data_reader.NUM_LABELS
        metric = RandomAblationCertifyMetric() 
        result_dicts = {"pix": []}
        for i in range(11):
            result_dicts[str(i)] = list()
        for data in description:
            target = self.data_reader.get_label_to_idx(data.label)
            data_length = data.length()
            keep_nums = data_length - round(data_length * args.sparse_mask_rate)

            tmp_instances = self.mask_instance_decorator(args, data, args.predict_ensemble)
            tmp_probs = predictor.predict_batch(tmp_instances)
            guess = np.argmax(np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels))

            if guess != target:
                metric(np.nan, data_length)
                continue

            numbers = args.ceritfy_ensemble * 2                
            tmp_instances, mask_indexes = self.mask_instance_decorator(args, data, numbers, return_indexes=True)
            ablation_indexes = [list(set(list(range(data_length))) - set(indexes)) for indexes in mask_indexes]
            tmp_probs = predictor.predict_batch(tmp_instances)
            tmp_preds = np.argmax(tmp_probs, axis=-1)
            p_i_x = np.bincount(tmp_preds, minlength=num_labels)[guess] / numbers
            result_dicts["pix"].append(p_i_x)
            for i in range(1, 11):
                lambda_value = population_lambda(tmp_preds, ablation_indexes, data_length, i, num_labels, guess)
                result_dicts[str(i)].append(lambda_value)
        
        file_name = os.path.join(args.logging_dir, "{}-probs.txt".format(args.build_logging_path()))
        with open(file_name, 'w') as file:
            for key, value in result_dicts.items():
                file.write(key)
                file.write(":  ")
                file.write(" ".join([str(v) for v in value]))
                file.write("\n")

    def saving_model_by_epoch(self, args: ClassifierArgs, epoch: int):
        # saving
        if args.saving_step is not None and args.saving_step != 0:
            if (epoch - 1) % args.saving_step == 0:
                self.save_model_to_file(args.saving_dir,
                                        args.build_saving_file_name(description='epoch{}'.format(epoch)))


    def mask_instance_decorator(self, args: ClassifierArgs, instance:InputInstance, numbers:int=1, return_indexes:bool=False):
        if self.forbidden_words is not None:
            forbidden_index = mask_forbidden_index(instance.perturbable_sentence(), self.forbidden_words)
            return mask_instance(instance, args.sparse_mask_rate, self.tokenizer.mask_token, numbers, return_indexes, forbidden_index)
        else:
            return mask_instance(instance, args.sparse_mask_rate, self.tokenizer.mask_token, numbers, return_indexes)

    # def sim(self, args: ClassifierArgs):
        # self.attack_csv_to_txt()
        # self.output_hidden()
        # self.output_sim
        

    @classmethod
    def run(cls, args: ClassifierArgs):
        # build logging
        # including check logging path, and set logging config
        args.build_logging_dir()
        args.build_logging()
        logging.info(args)

        args.build_environment()
        # check dataset and its path
        args.build_dataset_dir()

        args.build_saving_dir()
        args.build_caching_dir()

        if args.dataset_name in ['agnews', 'snli']:
            args.keep_sentiment_word = False

        classifier = cls(args)
        classifier.methods[args.mode](args)
