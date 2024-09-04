from transformers import BertPreTrainedModel, BertModel
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
import torch
import numpy


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
 
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
 
        self.init_weights()
        self.layer_weights = nn.Parameter(torch.ones(12))# 层权重初始化为全1
        self.softmax = nn.Softmax(dim=0)# 用于将权重归一化
 
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, output_hidden_states=True):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, 
                            output_hidden_states=output_hidden_states,)
        pooled_output_1 = outputs[1]
        hidden_states = outputs[2] # 元组13个tensor( 32, 59, 768)
        # 在前向传播中
        hidden_states = outputs[2][1:]  # 去掉输入嵌入层,只保留12个隐藏层
        last_layer = hidden_states[-1]
        
        # 获取第4-8层（索引为3-7）并计算平均
        middle_layers = hidden_states[1:4]  # 包含第4层到第8层
        middle_layers_mean = torch.mean(torch.stack(middle_layers), dim=0)

        # 对第12层和第4-8层的平均进行组合
        combined_output = (last_layer + middle_layers_mean) / 2

        # 对所有token取平均（除了第一个token，通常是[CLS]标记）
        pooled_output = torch.mean(combined_output[:, 1:, :], dim=1)

        #hidden_layer = 12
        #pooled_output_0 = torch.mean(hidden_states[hidden_layer][:,1:,:],dim=1) 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
 
        outputs = (logits,) + outputs[2:] # add hidden states and attention if they are here
 
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        layer_means = torch.stack([torch.mean(layer[:, 1:, :], dim=1) for layer in hidden_states])
 
        return outputs  # (loss), logits, (hidden_states), (attentions)