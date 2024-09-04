from operator import index, le
from tkinter import ttk
import numpy as np
from typing import List, Dict
from data.instance import InputInstance
from utils.mask_merge import panduan


def mask_forbidden_index(sentence: str, forbidden_words: List[str]) -> List[int]:
    sentence_in_list = sentence.split()
    forbidden_indexes = []
    for index, word in enumerate(sentence_in_list):
        if word in forbidden_words:
            forbidden_indexes.append(index)
    if len(forbidden_indexes) == 0:
        return None
    else:
        return forbidden_indexes

def sampling_index_loop_nums(length: int,
                             mask_numbers: int, 
                             add_indices: list,
                             nums: int, 
                             sampling_probs: List[float] = None) -> List[int]:
    if sampling_probs is not None:
        assert length == len(sampling_probs)
        if sum(sampling_probs) != 1.0:
            sampling_probs = sampling_probs / sum(sampling_probs)
    mask_indexes = []
    for x in range(nums):
        mask_index_temp = np.random.choice(list(range(length)), mask_numbers, replace=False, p=sampling_probs).tolist()
        attack = panduan(mask_index_temp,add_indices)
        index_test=0
        while(attack == 1):
            mask_index_temp = np.random.choice(list(range(length)), mask_numbers, replace=False, p=sampling_probs).tolist()
            attack = panduan(mask_index_temp,add_indices)
            index_test=index_test+1
            if(index_test==3 and x>0):
                mask_index_temp=mask_indexes[x-1]
                break
        mask_indexes.append(mask_index_temp)
    return mask_indexes

def panduan(mask_index_temp,add_indices):
    for i in mask_index_temp:
        for j in add_indices:
            if (i==j):
                return 1 
    return 0

def mask_instance(instance: InputInstance, 
                  rate: float, 
                  token: str, 
                  nums: int = 1, 
                  return_indexes: bool = False, 
                  forbidden_indexes: List[int] = None, 
                  random_probs: List[float] = None) -> List[InputInstance]:
    sentence = instance.perturbable_sentence()
    results = mask_sentence(sentence, rate, token, nums, return_indexes, forbidden_indexes, random_probs, unattack)
    if return_indexes:
        mask_sentences_list = results[0]
    else:
        mask_sentences_list = results
    tmp_instances = [InputInstance.from_instance_and_perturb_sentence(instance, sent) for sent in mask_sentences_list]
    if return_indexes:
        return tmp_instances, results[1]
    else:
        return tmp_instances

def xiabiao(s,un):
    add_indices=[]
    w1=s
    w2=un
    for i in range(min(len(w1),len(w2))):
        if w1[i]!=w2[i]:
            add_indices.append(i)
    return add_indices

def mask_sentence(sentence: str, 
                  unattack: str,
                  rate: float, 
                  token: str, 
                  nums: int = 1, 
                  return_indexes: bool = False, 
                  forbidden: List[int] = None,
                  random_probs: List[float] = None, 
                  min_keep: int = 2 ,
                   ) -> List[str]:
    # str --> List[str]
    sentence_in_list = sentence.split()
    unattack_in_list = unattack.split()
    length = len(sentence_in_list)
    length_1 = len(unattack_in_list)
    if(length != length_1):
        add_indices=[]
    else :
        add_indices=xiabiao(sentence_in_list,unattack_in_list)

    mask_numbers = round(length * rate)
    if length - mask_numbers < min_keep:
        mask_numbers = length - min_keep if length - min_keep >= 0 else 0
    tmp_sentences = []
    #k=1
    #if len(add_indices)==0:
        #tmp_sentences.append(unattack)
    #else:
        #for i in add_indices:
            #tmp_sentence = mask_sentence_by_indexes(sentence_in_list, i, token, forbidden)
            #tmp_sentences.append(tmp_sentence)

    #k=2
    """k=min(len(add_indices),mask_numbers)
    if( k==2 and len(add_indices)==2):
        tmp_sentence = mask_sentence_by_indexes(sentence_in_list, add_indices, token, forbidden)
        tmp_sentences.append(tmp_sentence)
    elif (k>2):
        tt_index=index_output(add_indices)
        for tt in tt_index:
            tmp_sentence = mask_sentence_by_indexes(sentence_in_list, tt, token, forbidden)
            tmp_sentences.append(tmp_sentence)

    else:
        tmp_sentences.append(unattack)"""

    #k=3
    """k=min(len(add_indices),mask_numbers)
    if( k==3 and len(add_indices)==3):
        tmp_sentence = mask_sentence_by_indexes(sentence_in_list, add_indices, token, forbidden)
        tmp_sentences.append(tmp_sentence)
    elif (k>3):
        tt_index=index_output(add_indices)
        for tt in tt_index:
            tmp_sentence = mask_sentence_by_indexes(sentence_in_list, tt, token, forbidden)
            tmp_sentences.append(tmp_sentence)
    else:
        tmp_sentences.append(unattack)"""

    #k=4
    k=min(len(add_indices),mask_numbers)
    if( k==4 and len(add_indices)==4):
        tmp_sentence = mask_sentence_by_indexes(sentence_in_list, add_indices, token, forbidden)
        tmp_sentences.append(tmp_sentence)
    elif (k>4):
        tt_index=index_output(add_indices,2)
        for tt in tt_index:
            tmp_sentence = mask_sentence_by_indexes(sentence_in_list, tt, token, forbidden)
            tmp_sentences.append(tmp_sentence)
    else:
        tmp_sentences.append(unattack)

    return tmp_sentences


def index_output(add_indices,x):
    t_indexs=[]
    for _ in range(x):
        t_indexs.append(np.random.choice(add_indices,4,replace=False))
    """for i in range(len(add_indices)):
        for j in range(i+1,len(add_indices)):
            for m in range(j+1,len(add_indices)):
                for n in range(m+1,len(add_indices)):
                    t_index=[]
                    t_index.append(add_indices[i])
                    t_index.append(add_indices[j])
                    t_index.append(add_indices[m])
                    t_index.append(add_indices[n])
                    t_indexs.append(t_index)"""
    return t_indexs


def mask_sentence_by_indexes(sentence: List[str], indexes, token: str, forbidden: List[str]=None) -> str:
    tmp_sentence = sentence.copy()
    for index in indexes:
        tmp_sentence[index] = token
    if forbidden is not None:
        for index in forbidden:
            tmp_sentence[index] = sentence[index]
    return ' '.join(tmp_sentence)