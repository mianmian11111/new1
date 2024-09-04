"""
HuggingFace Model Wrapper
--------------------------
"""
import os
from tokenizers import Tokenizer
import torch
import transformers
import random
import textattack
import numpy as np
import pandas as pd
import heapq
from .pytorch_model_wrapper import PyTorchModelWrapper

from typing import List, Tuple
from utils.mask import mask_sentence, mask_forbidden_index
from scipy.special import softmax
from sklearn.preprocessing import normalize
from utils.utils import build_forbidden_mask_words
from args import ClassifierArgs
from torch import nn as nn
from transformers import PreTrainedTokenizer, AutoModelForMaskedLM, RobertaTokenizer


class HuggingFaceModelMaskEnsembleWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""
    def __init__(self, 
                args: ClassifierArgs, 
                model: nn.Module, 
                tokenizer: PreTrainedTokenizer, 
                batch_size: int = 300, 
                with_lm: bool = False):
        self.model = model.to(textattack.shared.utils.device)
        self.mask_token = tokenizer.mask_token
        self.ensemble = args.predict_ensemble
        if isinstance(tokenizer, transformers.PreTrainedTokenizer):
            tokenizer = textattack.models.tokenizers.AutoTokenizer(tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dicttt=args.dicct
        self.mask_rate = args.sparse_mask_rate
        self.ensemble_method = args.ensemble_method
        self.max_seq_length = args.max_seq_length
        self.nums=args.nums

        self.forbidden_words = None
        if args.keep_sentiment_word:
            self.forbidden_words = build_forbidden_mask_words(args.sentiment_path)
        self.masked_lm = None
        if args.with_lm:
            self.lm_tokenizer = tokenizer.tokenizer
            self.masked_lm=self.model

    def _model_predict(self, inputs):
        """Turn a list of dicts into a dict of lists.

        Then make lists (values of dict) into tensors.
        """
        model_device = next(self.model.parameters()).device
        input_dict = {k: [_dict[k] for _dict in inputs] for k in inputs[0]}
        input_dict = {
            k: torch.tensor(v).to(model_device) for k, v in input_dict.items()
        }
        outputs = self.model(**input_dict)

        if isinstance(outputs[0], str):
            return outputs
        else:
            return outputs[0]

    def mask_sentence_decrator(self, sentence:str) -> List[str]:
        forbidden_index = None
        random_probs = None
        if self.forbidden_words is not None:
            forbidden_index = mask_forbidden_index(sentence, self.forbidden_words)
        if self.masked_lm is not None:
            random_probs = self.mask_probs(sentence)
        if random_probs:
            return mask_sentence(sentence, self.mask_rate, self.mask_token, self.ensemble, forbidden=forbidden_index, random_probs=random_probs),random_probs
        else:
            alist=[]
            azlist=[]
            alist.append(sentence) 
            azlist.append(random.randint(0,len(sentence)-1))
            return alist, azlist
 

    def mask_probs(self, sentence: str) -> List[int]:
        # encodings = self.tokenizer.tokenizer.encode_plus(sentence, truncation=True, max_length=self.max_seq_length, add_special_tokens=True, return_tensors='pt')
        # encodings = {key: value.cuda() for key, value in encodings.items()}
        # lm_logits = self.model.forward(encodings['input_ids'],encodings['attention_mask'])
        # labels=lm_logits.logits.argmax(dim=1)

        # a=self.wir(sentence)[0]
        # c=self.wdr(sentence)[0]
        # b=self.cipin(self.dicttt,sentence)
        # zz=list(set(c)|set(b))
        # return zz
        b=self.cipin(self.dicttt,sentence)
        return b


        #c=self.delete_words(sentence)[0]
        # index_=list(set(a)|set(b))

        # if b:
        #     index_=list(set(a)&set(b))
        #     if len(index_)==0:
        #         index_.extend(b)
        #         index_.append(a[0])
        #     if a[0] not in index_:
        #         index_.append(a[0])
        # else:
        #     index_=a[:2]
        # return  index_
        # if len(b)!=0 :
        #     if a[0] not in b :
        #         b.append(a[0])
        #         index_.extend(b)
        #     else:
        #         return b    
        # else:
        #     index_.extend(a[0:3])
        # return index_
        
    
    def delete_word_at_index_z(self,sentence):
        word_index_list=[]
        alist=sentence.split()
        len_text=len(alist)
        for i in range(len_text):
            c=alist.copy()
            if i==len_text-1:
                c=c[i-1]
                word_index_list.append(''.join(c))
            else:
                c=c[i+1:]
                word_index_list.append(' '.join(c))
        return word_index_list

    def delete_words(self,sentence):
        index_order=[]
        ids=self.encode([sentence])
        orig_preds = self._model_predict(ids)
        orig_label = orig_preds.argmax(dim=1)
        leave_one_texts=self.delete_word_at_index_z(sentence)
        scores=[]
        for i in leave_one_texts:
            result1=self.encode([i])
            predictions1=self._model_predict(result1)
            delete_labes=predictions1.argmax(dim=1)
            if orig_label==delete_labes:
               import_scores=orig_preds.max()-predictions1.max()
            else:
               import_scores=orig_preds.max()-predictions1.min()+predictions1.max()-orig_preds.min()
            scores.append(import_scores.item())
            orig_preds=predictions1
            orig_label=delete_labes
        #scores_im=self.Get_index(scores,5)
        sentence_list=sentence.split()
        if len(sentence_list)<5:
            scores_im=self.Get_index(scores,1)
        elif len(sentence_list)<20:
            scores_im=self.Get_index(scores,3)
        else:
            scores_im=self.Get_index(scores,5)
        scores_im=[x for x in scores_im]
        # _,scores_im=torch.topk(scores,5)
        index_order.append(scores_im)
        return index_order

    def cipin(self,dictt,sentence):
        attack_text=sentence.lower().split()
        list_index=[]
        for i in attack_text:
            if i  in dictt:
                list_index.append(dictt[i])
            else:
                list_index.append(0)
        list_cipin=self.Get_index_0_1(list_index)
        # return list_cipin
        return []

    def Get_index_0_1(self,list_):
        low_a=[
        idx 
        for idx ,w in enumerate(list_)
        # if w==0
        if w<=1
        ]
        return low_a
    def wir(self,sentence):
        index_order=[]
        ids=self.encode([sentence])
        orig_preds = self._model_predict(ids)
        orig_label = orig_preds.argmax(dim=1)
        leave_one_texts=self.delete_word_at_index(sentence)
        scores=[]
        for i in leave_one_texts:
            result1=self.encode([i])
            predictions1=self._model_predict(result1)
            delete_labes=predictions1.argmax(dim=1)
            if orig_label==delete_labes:
               import_scores=orig_preds.max()-predictions1.max()
            else:
               import_scores=orig_preds.max()-predictions1.min()+predictions1.max()-orig_preds.min()
            scores.append(import_scores.item())
        #scores_im=self.Get_index(scores,5)
        sentence_list=sentence.split()
        if len(sentence_list)<5:
            scores_im=self.Get_index(scores,2)
        elif len(sentence_list)<20:
            scores_im=self.Get_index(scores,3)
        else:
            scores_im=self.Get_index(scores,3)
        scores_im=[x for x in scores_im]
        # _,scores_im=torch.topk(scores,5)
        index_order.append(scores_im)
        return index_order

    def wdr(self,sentence):
        index_order=[]
        ids=self.encode([sentence])
        orig_preds = self._model_predict(ids)
        orig_label = orig_preds.argmax(dim=1)
        leave_one_texts=self.delete_word_at_index(sentence)
        scores=[]
        for i in leave_one_texts:
            result1=self.encode([i])
            predictions1=self._model_predict(result1)
            delete_labes=predictions1.argmax(dim=1)
            if orig_label==delete_labes:
               import_scores=predictions1.max()-predictions1.min()
            else:
               import_scores=predictions1.min()-predictions1.max()
            scores.append(import_scores.item())
        #scores_im=self.Get_index(scores,5)
        sentence_list=sentence.split()
        if len(sentence_list)<20:
            scores_im=self.Get_index1(scores,2)
        else:
            scores_im=self.Get_index1(scores,4)
        scores_im=[x for x in scores_im]
        # _,scores_im=torch.topk(scores,5)
        index_order.append(scores_im)
        return index_order    


    def delete_word_at_index(self,initial_text):
        word_index_list=[]
        alist=initial_text.split()
        len_text=len(alist)
        for i in range(len_text):
            c=alist.copy()
            c.pop(i)
            word_index_list.append(' '.join(c))
        return word_index_list
    def Get_index(self,list_,n):
        N_la=pd.DataFrame({'score':list_}).sort_values(by='score',ascending=[False])
        return list(N_la.index)[:n]

    def Get_index1(self,list_,n):
        return list(map(list_.index, heapq.nsmallest(n, list_)))

    def mask_lm_loss(self, ids: torch.Tensor, pred: torch.Tensor, delete_special_tokens:bool=True) -> torch.Tensor:
        loss = torch.nn.functional.cross_entropy(pred, ids, reduction='none')
        if delete_special_tokens:
            loss[ids == self.lm_tokenizer.pad_token_id] = 0.0
            loss[ids == self.lm_tokenizer.sep_token_id] = 0.0
            loss[ids == self.lm_tokenizer.cls_token_id] = 0.0
        return loss

    def get_tokenizer_mapping_for_sentence(self, sentence: str) -> Tuple: 
        if isinstance(self.lm_tokenizer, RobertaTokenizer):
            sentence_tokens = sentence.split()
            enc_result = [self.lm_tokenizer.encode(sentence_tokens[0], add_special_tokens=False)]
            enc_result.extend([self.lm_tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) for x in sentence_tokens[1:]])
        else:
            enc_result = [self.lm_tokenizer.encode(x, add_special_tokens=False) for x in sentence.split()]
        desired_output = []
        idx = 1
        for token in enc_result:
            tokenoutput = []
            for _ in token:
                tokenoutput.append(idx)
                idx += 1
            desired_output.append(tokenoutput)
        return (enc_result, desired_output)

    def get_word_loss(self, indexes: List[int], losses: torch.Tensor) -> float:
        try:
            loss = []
            for index in indexes:
                if index <= self.max_seq_length - 2:
                    loss.append(losses[index].item())
                else:
                    loss.append(0.0)
            return np.mean(loss)
        except:
            return 0.0

    # def mask_probs(self, sentence: str) -> List[int]:
    #     # encodings = self.tokenizer.tokenizer.encode_plus(sentence, truncation=True, max_length=self.max_seq_length, add_special_tokens=True, return_tensors='pt')
    #     encodings = self.tokenizer.tokenizer.encode_plus(sentence, truncation=True, max_length=self.max_seq_length, add_special_tokens=True, return_tensors='pt')
    #     encodings = {key: value.cuda() for key, value in encodings.items()}
    #     lm_logits = self.model.forward(encodings['input_ids'],encodings['attention_mask'])
    #     labels=lm_logits.logits.argmax(dim=1)
    #     #indice_list=[]
    #     # delta_grad_=self.ml_get_emb_grad(sentence)['gradient']
    #     # delta_grad_=self.get_emb_grad(sentence)
    #     delta_grad_=self.get_emb_grad(encodings['input_ids'],encodings['attention_mask'],labels)
    #     delta_grad = delta_grad_[0].detach()
    #     norm_grad=torch.norm(delta_grad,p=2,dim=-1)
    #     # 这个代码存在问题（应该是句子大小+cls和sep）
        
    #     items=norm_grad.shape[1]
    #     #print(norm_grad[0,:items])
    #     if len(sentence.split()) < 5:
    #         _,indice_=torch.topk(norm_grad[0,:items],1)
    #         indice=[indice_.item()]
    #     else:
    #         _,indice_=torch.topk(norm_grad[0,:items],self.k)
    #         indice=[x.item() for x in indice_]
    #     # indice_list.append(indice)
    #     return  indice

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        mask_text_input_list = []
        if isinstance(text_input_list[0], tuple) and len(text_input_list[0]) == 2:
            for text_input in text_input_list:
                mask_text_input = [(text_input[0], sentence) for sentence in self.mask_sentence_decrator(text_input[1])]
                mask_text_input_list.extend(mask_text_input)
        else:
            if self.nums==0 or self.nums==1:
                mask_text_input_list.extend(text_input_list)
                self.nums=self.nums+1
            else:
                self.nums=self.nums+1
                mask_index=[]
                for text_input in text_input_list:
                        zz=self.mask_sentence_decrator(text_input)
                        mask_text_input_list.extend(zz[0])
                        mask_index.append(zz[1])
        # else:
        #     for text_input in text_input_list:
        #         mask_text_input_list.extend(self.mask_sentence_decrator(text_input))
        ids = self.encode(mask_text_input_list)  
        with torch.no_grad():
            outputs = textattack.shared.utils.batch_model_predict(
                self._model_predict, ids, batch_size=self.batch_size
            )
        label_nums = outputs.shape[1]
        # if self.nums>2:
        #     z2=[]
        #     z1=0
        #     for i in range(len(mask_index)):
        #         z1+=len(mask_index[i])
        #         z2.append(z1)
        #     ensemble_logits_for_each_input=np.split(outputs, z2, axis=0)
        #     ensemble_logits_for_each_input.pop()
        # else:
        #     ensemble_logits_for_each_input = np.split(outputs, indices_or_sections=len(text_input_list), axis=0)
        #single
        ensemble_logits_for_each_input = np.split(outputs, indices_or_sections=len(text_input_list), axis=0)
        logits_list = []
        for logits in ensemble_logits_for_each_input:
            if self.ensemble_method == 'votes':
                probs = np.bincount(np.argmax(logits, axis=-1), minlength=label_nums) / len(logits)
                logits_list.append(np.expand_dims(probs, axis=0))
            else:
                probs = normalize(logits, axis=1)
                probs = np.mean(probs, axis=0, keepdims=True)
                logits_list.append(probs)

        outputs = np.concatenate(logits_list, axis=0)
        return outputs
        #ensemble_logits_for_each_input = np.split(outputs, indices_or_sections=len(text_input_list), axis=0)
        
        #logits_list = []
        # for logits in ensemble_logits_for_each_input:
        #     if self.ensemble_method == 'votes':
        #         probs = np.bincount(np.argmax(logits, axis=-1), minlength=label_nums) /len(logits)
        #         logits_list.append(np.expand_dims(probs, axis=0))
        #     else:
        #         probs = normalize(logits, axis=1)
        #         probs = np.mean(probs, axis=0, keepdims=True)
        #         logits_list.append(probs)

        if self.ensemble_method == 'votes':
            probs = np.bincount(np.argmax(outputs, axis=-1), minlength=label_nums)/len(outputs)
            outputs[0][0]=probs[0]
            outputs[0][1]=probs[1]
            if len(outputs)==1:
                return outputs
            else:
                for i in range(1,len(outputs)):
                    outputs[i]=outputs[0]
        # else:
        #     probs = normalize(outputs, axis=1)
        #     probs = np.mean(probs, axis=0, keepdims=True)
        #     logits_list.append(probs)

        return outputs

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids = self.encode([text_input])
        predictions = self._model_predict(ids)

        model_device = next(self.model.parameters()).device
        input_dict = {k: [_dict[k] for _dict in ids] for k in ids[0]}
        input_dict = {
            k: torch.tensor(v).to(model_device) for k, v in input_dict.items()
        }
        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0]["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(x)["input_ids"])
            for x in inputs
        ]

    def get_emb_grad(self,input_ids,attention_mask,labels):
        self.model.eval()
        #embedding_layer = self.masked_lm.get_input_embeddings()
        embedding_layer = self.model.roberta.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True
        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])
        emb_hook = embedding_layer.register_full_backward_hook(grad_hook)
        self.model.zero_grad()
        ouput=self.model.forward(input_ids,attention_mask,labels=labels)
        loss=ouput['loss']
        
        ##
        # encodings = self.lm_tokenizer.encode_plus(text_input, truncation=True, max_length=self.max_seq_length, add_special_tokens=True, return_tensors='pt')
        # encodings = {key: value.cuda() for key, value in encodings.items()}
        # lm_logits = self.masked_lm(**encodings)[0]
        # labels=lm_logits.argmax(dim=1)
        # loss_fct =nn.CrossEntropyLoss()
        # loss = loss_fct(lm_logits.view(-1, 2), labels.view(-1))
        #lm_losses = self.masked_lm(encodings["input_ids"][0], attention_mask=encodings["attention_mask"][0],labels=labels)[0]



       # ids = self.encode([text_input])
       # predictions = self._model_predict(ids)
        # model_device = next(self.model.parameters()).device
        # input_dict = {k: [_dict[k] for _dict in ids] for k in ids[0]}
        # input_dict = {
        #     k: torch.tensor(v).to(model_device) for k, v in input_dict.items()
        # }
        # labels = predictions.argmax(dim=1)
        # loss = self.model(**input_dict, labels=labels)[0]
        loss.backward()
        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        return emb_grads

    def ml_get_emb_grad(self,text_input):
        self.masked_lm.eval()
        embedding_layer = self.masked_lm.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_full_backward_hook(grad_hook)

        self.masked_lm.zero_grad()
        model_device = next(self.masked_lm.parameters()).device
        ids = self.encode([text_input])
        predictions = self._model_predict(ids)
        model_device = next(self.masked_lm.parameters()).device
        input_dict = {k: [_dict[k] for _dict in ids] for k in ids[0]}
        input_dict = {
            k: torch.tensor(v).to(model_device) for k, v in input_dict.items()
        }
        try:
            labels = predictions.argmax(dim=1)
            loss = self.masked_lm(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.masked_lm)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )
        loss.backward()
        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()

        output = {"ids": ids[0]["input_ids"], "gradient": emb_grads}
        return output