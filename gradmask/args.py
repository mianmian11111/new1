import os
import torch
import logging
import argparse
import pandas as pd
from overrides import overrides
# from utils.config import DATASET_TYPE 
# from utils.config import PRETRAINED_MODEL_TYPE
from utils import check_and_create_path, set_seed
from baseargs import ProgramArgs

class ClassifierArgs(ProgramArgs):
    def __init__(self):
        super(ClassifierArgs, self).__init__()
        self.mode = 'train'  # in ['train', 'attack', 'evaluate', 'certify', 'statistics']
        self.file_name = None   #  filename of the saved model, default is None. See build_saving_file_name()

        self.seed = 42    # for seed of all random function(including, np.random, random, torch, so on.)
        self.evaluation_data_type = 'test'  # default is 'test', meaning using test set to evaluate
        self.attacked_data_type = ['train', 'test']
        self.attack_data_type = 'test'
        self.attack = 'textbugger'
        self.attack_list = ['textbugger','textfooler','deepwordbug']
        self.sim_f = ['cosine_similarity','点积','欧式距离','皮尔逊相关系数','曼哈顿距离']
        self.layer_list = [0,1,2,3,4,5,6,7,8,9,10,11]
        #self.hidden_data_type = ['org_test','att_test','org_train','att_train'] 
        self.hidden_data_type = ['org_test','att_test','org_train','org_train'] 
        self.data_rate = 1


	    # 表示best_model的后缀，代表当前是哪个模型
        self.tag = 12
        self.layer = 12
        # 是否使用微调后的模型参数
        self.parameter_fine_tuning = True
        self.classifier = 'MLP'
        self.pred_data_type = ['att_test_succ','att_train_succ','org_test_succ','org_train_succ']
        

        self.dataset_name = 'yelp'
        self.dataset_dir = '/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/dataset'
        self.model_type = 'bert'
        self.model_name_or_path = '/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/bert-model/bert-base-uncased'
        #self.model_type = 'bert-sst2'
        #self.model_name_or_path = '/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/bert-base-uncased-SST-2'
        # self.model_type = 'roberta'
        # self.model_name_or_path = 'roberta-base'
        self.k_nums=1

        self.max_seq_length = 256       # the maximum length of input text to be truncated, for imdb, recommends 256; else, 128
        self.do_lower_case = True

        self.epochs = 10  # training epochs
        self.batch_size = 64# batch size defaut:32
        self.gradient_accumulation_steps = 1  # Number of updates steps to accumulate before performing a backward/update pass.
        self.learning_rate = 5e-5  # The initial learning rate for Adam.
        self.weight_decay = 1e-6  # weight decay
        self.adam_epsilon = 1e-8  # epsilon for Adam optimizer
        self.max_grad_norm = 1.0  # max gradient norm
        self.learning_rate_decay = 0.1  # Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training

        self.compare_key = '+accuracy'

        # saving, logging and caching dir
        self.caching_dir = '/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/cache_path'  # for cache path
        self.saving_dir = '/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/save_models'  # for saving modeling
        self.logging_dir = '/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/result_log'  # for logging
        self.saving_step = 0  # saving step for epoch

        self.tensorboard = None

        # the type to train models, only support 'freelb',  'pgd',  'hotflip', 'sparse', 'safer' now
        # See build_trainer in classifier.py
        # default is '', meaning normal training
        # freelb: FreeLB: Enhanced adversarial training for natural language understanding
        # pgd: Towards deep learning models resistant to adversarial attacks.
        # hotflip: HotFlip: White-box adversarial examples for text classification
        # safer: A structure-free approach for certified robustness to adversarial word substitutions. 
        # sparse: for RanMASK. Certified Robustness to Text Adversarial Attacks by Randomized [MASK]
        self.training_type = ''

        # for RanMASK
        self.sparse_mask_rate = 0.3# the mask rate for RanMASK
        self.predict_ensemble = 1 # the ensemble number for random smoothing (using for attack and predict)
        self.predict_numbers = 1000 # Not used
        # for certify on RanMASK
        self.ceritfy_ensemble = 5000 # the ensemble number for random smoothing (using for certify). It is much larger than predict_ensemble, but slower.
        self.certify_numbers = 1000 # the dataset size to certify. default is None, meaning all dataset is used to evaluate
        # whether to add lambda, for sparse NLP, 
        # pi(x)− Pr([f(ABLATE(x, T)) = i] ∧ [T ∩ (x diff with x')]) <= pi(x')
        # Pr([f(ABLATE(x, T)) = i] ∧ [T ∩ (x diff with x') != Ø]) = Pr([f(ABLATE(x, T))
        # where lambda = Pr([f(ABLATE(x, T)) = i] | [T ∩ (x diff with x')  != Ø]) * Pr(T ∩ (x diff with x')  != Ø)
        # if lambda is False, only use the delta Pr(T ∩ (x diff with x')  != Ø) for certificate robustness 
        self.certify_lambda = True
        # for confidence alpha probability
        self.alpha = 0.05
        # whether to use language model (lm) to decide the masking indexes when attacking. For empirical robustness, using lm is better.
        self.with_lm = True

        # for pgd-K and FreeLB (including adv-hotflip)
        self.adv_steps = 5 # Number of gradient ascent steps for the adversary, for FreeLB default 5
        self.adv_learning_rate = 3e-2 # Step size of gradient ascent, for FreeLB, default 0.03
        self.adv_init_mag = 5e-2 # Magnitude of initial (adversarial?) perturbation, for FreeLB, default 0.05
        self.adv_max_norm = 0.0 # adv_max_norm = 0 means unlimited, for FreeLB, default 0.0
        self.adv_norm_type = 'l2' # norm type of the adversary
        self.adv_change_rate = 0.2 # rate for adv-hotflip, change rate of a sentence

        # for SAFER
        self.safer_perturbation_set = 'perturbation_constraint_pca0.8_100.pkl'   # perturbation set path for safer trainer

        # for attack
        self.attack_times = 1 # attack times for average record
        self.attack_method = 'deepwordbug' # attack algorithm
        self.attack_numbers = 1000 # the examples numbers to be attack
        self.ensemble_method = 'votes' # in [votes mean], the ensemble type, Ses RanMASK paper

        # The following are some tricks that have been tried and used for training and can be ignored
        # for sentiment-word file path
        self.sentiment_path = '/root/xuce/RanMaskT1/RanMASK/dataset/sentiment_word/sentiment-words.txt'
        self.keep_sentiment_word = False
        self.incremental_trick = False
        self.initial_mask_rate = 0.4
        self.saving_last_epoch = False
        self.train='/share/home/u2315363122/MI4D/mi4d-j/mi4d-j/gradmask/dataset/yelp/train_cp.tsv'
        self.dicct=self.cp()
        self.nums=0
        
    def cp(self):
        dictt={}
        train=pd.read_csv(self.train,sep='\t')
        alist=train['sentence']
        z=alist.tolist()
        for i in z:
            i=i.lower().split()
            for j in i:
                if j not in dictt:
                    dictt[j]=1
                else :
                    z=dictt[j]+1
                    dictt[j]=z
        return dictt

    def build_environment(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.seed)

    def build_dataset_dir(self):
        #assert self.dataset_name in DATASET_TYPE.DATA_READER.keys(), 'dataset not found {}'.format(self.dataset_name)
        testing_file = ['train.json', 'train.txt', 'train.csv', 'train.tsv']
        for file in testing_file:
            train_file_path = os.path.join(self.dataset_dir, file)
            if os.path.exists(train_file_path) and os.path.isfile(train_file_path):
                print(f"Found dataset file: {train_file_path}")
                return
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)
        for file in testing_file:
            train_file_path = os.path.join(self.dataset_dir, file)
            if os.path.exists(train_file_path) and os.path.isfile(train_file_path):
                print(f"Found dataset file: {train_file_path}")
                return
        raise FileNotFoundError("Dataset file cannot be found in dir {}".format(self.dataset_dir))

    def build_saving_dir(self):
        self.saving_dir = os.path.join(self.saving_dir,  "{}_{}".format(self.dataset_name, self.model_type))
        check_and_create_path(self.saving_dir)

    def build_logging_dir(self):
        self.logging_dir = os.path.join(self.logging_dir, "{}_{}".format(self.dataset_name, self.model_type))
        check_and_create_path(self.logging_dir)

    def build_caching_dir(self):
        if self.safer_perturbation_set is not None:
            self.safer_perturbation_set = os.path.join(self.caching_dir, os.path.join(self.dataset_name, self.safer_perturbation_set))
        self.caching_dir = os.path.join(self.caching_dir, "{}_{}".format(self.dataset_name, self.model_type))
        check_and_create_path(self.caching_dir)

    def build_logging(self, **kwargs):
        logging_file_path = self.build_logging_file()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=logging_file_path,level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    def build_saving_file_name(self, description: str = None):
        file_name = self.training_type
        if self.file_name is not None:
            file_name = "{}{}".format(file_name if file_name == "" else file_name+"_", self.file_name) 
        hyper_parameter_dict = {'len': self.max_seq_length, 'epo': self.epochs, 'batch': self.batch_size}
        if self.training_type == 'freelb' or self.training_type == 'pgd':
            hyper_parameter_dict['advstep'] = self.adv_steps
            hyper_parameter_dict['advlr'] = self.adv_learning_rate
            hyper_parameter_dict['norm'] = self.adv_max_norm
        elif self.training_type == 'advhotflip':
            hyper_parameter_dict['rate'] = self.adv_change_rate
            hyper_parameter_dict['advstep'] = self.adv_steps
        if self.training_type == 'sparse':
            hyper_parameter_dict['rate'] = self.sparse_mask_rate

        if file_name == "":
            file_name = '{}'.format("-".join(["{}{}".format(key, value) for key, value in hyper_parameter_dict.items()]))
        else:
            file_name = '{}-{}'.format(file_name, "-".join(["{}{}".format(key, value) for key, value in hyper_parameter_dict.items()]))

        if description is not None:
            file_name = '{}-{}'.format(file_name, description)
        return file_name

    def build_logging_path(self):
        if self.mode is None:
            return self.build_saving_file_name()
        if self.mode == 'certify' and self.certify_lambda:
            return '{}-{}-{}'.format(self.mode, self.build_saving_file_name(), 'lambda')
        elif self.mode == 'attack':
            if self.training_type in ['sparse', 'safer']:
                logging_path = "{}-{}-{}".format(self.mode, self.build_saving_file_name(), self.ensemble_method)
                if self.with_lm:
                    logging_path = "{}-{}".format(logging_path, 'lm')
            else:
                logging_path = "{}-{}".format(self.mode, self.build_saving_file_name())
            return logging_path
        else:
            return '{}-{}'.format(self.mode, self.build_saving_file_name())

    def build_logging_file(self):
        if self.mode == 'attack':
            logging_path = self.build_logging_path()
            logging_path = os.path.join(self.logging_dir, logging_path)
            if not os.path.exists(logging_path):
                os.makedirs(logging_path)
            return os.path.join(logging_path, 'running.log')
        else:
            return os.path.join(self.logging_dir, '{}.log'.format(self.build_logging_path()))
