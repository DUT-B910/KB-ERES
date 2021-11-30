# -*- coding: utf-8 -*-

'''
Filename : config.py
Function : All the file path for input and output, users could change them for usage.
           All the parameter settings used in KR-ERE, users could change any as they need.
'''

import torch
import os

def check_path(path, raise_error=True):
    if not os.path.exists(path):
        if raise_error:
            print("error : Path %s does not exist!" % path)
            exit(1)
        else:
            print("warning : Path %s does not exist!" % path)
            print("info : Creating path %s." % path)
            os.makedirs(path)
            print("info : Successfully making dir!")

class Config():
    def __init__(self):
        # Input&Output data arguments
        self.ori_train_path = './data/original/TrainingSet.txt'
        self.ori_dev_path   = './data/original/DevelopmentSet.txt'
        self.ori_test_path  = './data/original/TestSet.txt'

        self.intra_path = './data/process/intra_ins/'
        self.inter_path = './data/process/inter_ins/'

        self.train_ins_path = 'TrainingSet.instance'
        self.dev_ins_path   = 'DevelopmentSet.instance'
        self.test_ins_path  = 'TestSet.instance'

        self.clean_data = 'data.pkl'
        self.word_vec_path = '' # 'word2vec.vec'

        self.entity_index_path    = './data/knowledge_base/entity2id.txt'
        self.relation_index_path  = './data/knowledge_base/relation2id.txt'
        self.triple_path          = './data/knowledge_base/train.txt'
        self.entity_vec_path      = './data/knowledge_base/entity2vec.bern'
        self.relation_vec_path    = './data/knowledge_base/relation2vec.bern'

        self.result_path = './result/KR_ERE/'
        self.intra = 'intra/'
        self.inter = 'inter/'
        self.document = 'merge/'
        self.pretrain_model_path = ''
        self.model_save_path0 = './result/KR_ERE/model/'
        self.model_save_path = self.model_save_path0 + 'KR_ERE_model.param'

        # Hyper-parameter settings
        self.word_vec_dim = 100
        self.kg_vec_dim = 100
        self.intra_lr = 0.0001
        self.inter_lr = 0.0002
        self.class_number = 2
        self.epoch_number = 30
        self.batch_size = 20
        self.convolution_dim = 100
        self.kernel_size = [1,2,3,4,5]
        self.is_document = True

        # Check Path
        self.CheckPath()

    def CheckPath(self):
        # Check files
        check_path(self.ori_train_path)
        check_path(self.ori_dev_path)
        check_path(self.ori_test_path)
        check_path(self.ori_train_path)
        check_path(self.entity_index_path)
        check_path(self.relation_index_path)
        check_path(self.triple_path)

        # Check dirs
        check_path(self.intra_path, raise_error=False)
        check_path(self.inter_path, raise_error=False)
        check_path(self.result_path + self.intra, raise_error=False)
        check_path(self.result_path + self.inter, raise_error=False)
        check_path(self.result_path + self.document, raise_error=False)
        check_path(self.model_save_path0, raise_error=False)


