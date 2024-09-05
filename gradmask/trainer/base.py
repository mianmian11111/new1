import logging

import torch
from tqdm import tqdm
from typing import Tuple
import torch.nn as nn
from args import ClassifierArgs
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from utils.utils import convert_batch_to_bert_input_dict


class BaseTrainer:
    def __init__(self,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        self.data_loader = data_loader
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.global_step = 0
        # 添加可学习参数weight，用于调整每一层的重要性
        self.layer_weight = self.model.layer_weights# 层权重初始化为全1
        self.softmax = nn.Softmax(dim=0)# 用于将权重归一化

    def train_epoch(self, args: ClassifierArgs, epoch: int) -> None:
        print("Epoch {}:".format(epoch))
        logging.info("Epoch {}:".format(epoch))
        self.model.train()

        epoch_iterator = tqdm(self.data_loader)
        oom_number = 0
        for batch in epoch_iterator:
            try:
                loss = self.train_batch(args, batch)
                epoch_iterator.set_description('loss: {:.4f}'.format(loss))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.warning('oom in batch forward / backward pass, attempting to recover from OOM')
                    print('oom in batch forward / backward pass, attempting to recover from OOM')
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    oom_number += 1
                else:
                    raise e
        logging.warning('oom number : {}, oom rate : {:.2f}%'.format(oom_number, oom_number / len(self.data_loader) * 100))
        return

    def train_batch(self, args: ClassifierArgs, batch: Tuple) -> float:
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss = self.train(args, batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.write_tensorboard(loss)
        self.global_step += 1
        return loss

    def train(self, args: ClassifierArgs, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.cuda() for t in batch)
        logits = self.forward(args, batch)[0]
        golds = batch[3]
        
        # 对权重进行softmax,确保它们的和为1
        normalized_weights  = self.softmax(self.layer_weight)
        
        # 假设 logits 是一个列表，其中包含了每一层的 logits,算每一层的损失加起来的总的损失
        total_loss = 0.00
        layer_losses = []
        for i, layer_logits in enumerate(logits):
            # 计算当前层的损失，loss_function返回每个样本的损失值向量
            layer_loss = self.loss_function(layer_logits, golds.view(-1))
            # 损失值向量算平均，得到一个平均的损失值
            layer_loss = torch.mean(layer_loss)
            # 将损失乘以相应的权重
            layer_loss = layer_loss * normalized_weights[i]
            # 将损失添加到总损失中
            total_loss += layer_loss
            # 记录每一层的损失，用于调试或分析
            layer_losses.append(layer_loss)
        
        # losses = self.loss_function(logits, golds.view(-1))
        # loss = torch.mean(losses)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        
        print("layer_weights:", self.layer_weight)
        
        # 打印层损失
        # 打印或记录每层的损失
        for i, layer_loss in enumerate(layer_losses):
            # tem() 方法用于将一个包含单个元素的张量（tensor）转换为一个Python数值（如int或float）。
            print(f"Layer {i} loss: {layer_loss.item()}")
        return total_loss.item()

    def 原_train(self, args: ClassifierArgs, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.cuda() for t in batch)
        logits = self.forward(args, batch)[0]
        golds = batch[3]
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def forward(self, args: ClassifierArgs, batch: Tuple) -> Tuple:
        '''
        for Bert-like model, batch_input should contains "input_ids", "attention_mask","token_type_ids" and so on
        '''
        inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
        return self.model(**inputs)

    def write_tensorboard(self, loss: float, **kwargs):
        # if self.writer is not None:
        #     self.writer.add_scalar('loss', loss, self.global_step)
        pass
