# -*- coding: utf-8 -*-
'''
Filename : main.py
Function : Train and test the model
           1. Train and test model on intra-sentence instances
           2. Train and test model on inter-sentence instances
           3. Merge the best results (on development set) of intra- and inter-sentence instances
'''

import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from merge_result import merge_intra_inter
import torch.nn.functional as F
from config import Config
from KR_ERE_model import KR_ERE

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
mySeed = np.random.RandomState(1234)
config = Config()

def get_property(max_len,wordEmbed,entityEmbed,relationEmbed,sentence,ent_feat,triple):
    sent_id = ent_feat[0]
    e1_id = ent_feat[1]
    e1_start_pos = ent_feat[2]
    e1_end_pos   = ent_feat[3]
    e2_id = ent_feat[4]
    e2_start_pos = ent_feat[5]
    e2_end_pos   = ent_feat[6]
    len_e1 = e1_end_pos - e1_start_pos
    len_e2 = e2_end_pos - e2_start_pos
    ######################generate the sequence##################################
    e1 = sentence[e1_start_pos:e1_end_pos]
    e2 = sentence[e2_start_pos:e2_end_pos]
    pos = sorted([(e1_start_pos,-1),(e1_end_pos,-1),(e2_start_pos,-2),(e2_end_pos,-2)],key=lambda p : p[0])
    sentence = sentence[:pos[0][0]]+[pos[0][1]]+sentence[pos[1][0]:pos[2][0]]+[pos[2][1]]+sentence[pos[3][0]:]
    n        = len(sentence)

    e1vectors = torch.cat([wordEmbed(Variable(torch.LongTensor([int(e)]).cuda())).view(100,1) for e in e1],1)
    e2vectors = torch.cat([wordEmbed(Variable(torch.LongTensor([int(e)]).cuda())).view(100,1) for e in e2],1)

    e1vector      = torch.sum(e1vectors,1)/len_e1
    e2vector      = torch.sum(e2vectors,1)/len_e2
    wordVectorLength = len(e2vector)
    #wordsembeddings:
    
    if sentence[0] == -1:
        words = e1vector.view(wordVectorLength,1)
    elif sentence[0] == -2:
        words = e2vector.view(wordVectorLength,1)
    else:
        words = wordEmbed(Variable(torch.LongTensor([sentence[0]]).cuda())).view(wordVectorLength,1)
    for word in sentence[1:]:
        if word == -1:
            words = torch.cat([words, e1vector.view(wordVectorLength,1)],1)
        elif word == -2:
            words = torch.cat([words, e2vector.view(wordVectorLength,1)],1)
        else:
            words = torch.cat([words, wordEmbed(Variable(torch.LongTensor([word]).cuda())).view(wordVectorLength,1)],1)
    wordswithPos = words
    ##########################generate the kb##############################################
    E1 = entityEmbed(Variable(torch.LongTensor([triple[0]]).cuda()))
    E2 = entityEmbed(Variable(torch.LongTensor([triple[1]]).cuda()))
    relation = relationEmbed(Variable(torch.LongTensor([triple[2]]).cuda()))
    return sent_id, wordswithPos, e1_id,e2_id,E1,E2,e1vectors,e2vectors,len_e1,len_e2,relation, n


def train_and_test(data_path,save_path,is_intra,func):
    #####################load the data###########################
    
    train_Sentence,train_Label,train_Entities,train_Triples,train_gold_sum,\
    dev_Sentence,dev_Label,dev_Entities,dev_Triples,dev_gold_sum,\
    test_Sentence,test_Label,test_Entities,test_Triples,test_gold_sum,\
    word2vector,entity2vector,relation2vector = pickle.load(open(data_path,'rb'),encoding = 'iso-8859-1') #all the data for model training
    #####################get he feature###########################sentence length
    max_len = max(max([len(sentence) for sentence in train_Sentence]),\
                  max([len(sentence) for sentence in dev_Sentence]),\
                  max([len(sentence) for sentence in test_Sentence]))
    max_word_num = len(word2vector)  #the number of the words
    embedding_dim = len(word2vector[0])
    embedding_dim_kg = len(entity2vector[0])
    print ("max sentence length: {}".format(max_len))
    print ("unique word number: {}".format(max_word_num))
    print ("word embedding dim: {}".format(embedding_dim))
    print ("knowledge embedding dim: {}".format(embedding_dim_kg))
    #################prepare the embedding layer#################

    word2vector = nn.Parameter(torch.FloatTensor(word2vector).cuda())
    wordEmbed = nn.Embedding(max_word_num,embedding_dim)
    wordEmbed.weight = word2vector

    entitynum = len(entity2vector)
    entity2vector = nn.Parameter(torch.FloatTensor(entity2vector).cuda())
    entityEmbed = nn.Embedding(entitynum,embedding_dim_kg)
    entityEmbed.weight = entity2vector

    relationnum = len(relation2vector)
    relation2vector = nn.Parameter(torch.FloatTensor(relation2vector).cuda())
    relationEmbed = nn.Embedding(relationnum,embedding_dim_kg)
    relationEmbed.weight = relation2vector

    ######################prepare the data#######################
    print ('generate the train data...')
    train_set = []
    for i in range(len(train_Sentence)):
        sampleTuple = get_property(max_len = max_len,
                                   wordEmbed = wordEmbed,
                                   entityEmbed = entityEmbed,
                                   relationEmbed = relationEmbed,
                                   sentence = train_Sentence[i],
                                   ent_feat = train_Entities[i],
                                   triple =train_Triples[i])
        train_set.append(sampleTuple)
    print ('generate the development data...')
    dev_set = []
    for i in range(len(dev_Sentence)):
        sampleTuple = get_property(max_len = max_len,
                                   wordEmbed = wordEmbed,
                                   entityEmbed = entityEmbed,
                                   relationEmbed = relationEmbed,
                                   sentence = dev_Sentence[i],
                                   ent_feat = dev_Entities[i],
                                   triple =dev_Triples[i])
        dev_set.append(sampleTuple)

    print ('generate the test data...')
    test_set = []
    for i in range(len(test_Sentence)):
        sampleTuple = get_property(max_len = max_len,
                                   wordEmbed = wordEmbed,
                                   entityEmbed = entityEmbed,
                                   relationEmbed = relationEmbed,
                                   sentence = test_Sentence[i],
                                   ent_feat = test_Entities[i],
                                   triple =test_Triples[i])
        test_set.append(sampleTuple)
    #########################Model###############################    
    model = func(wordEmbed,entityEmbed,relationEmbed)
    print("model fitting - {}".format(model.name))
    prediction = model.train_fit(trainset = train_set,
                                 trainLabel = train_Label,
                                 train_gold_num = train_gold_sum,
                                 valset = dev_set,
                                 valLabel = dev_Label,
                                 val_gold_num = dev_gold_sum,
                                 testset = test_set,
                                 testLabel = test_Label,
                                 test_gold_num = test_gold_sum,
                                 resultOutput = save_path,
                                 is_intra = is_intra,
                                 pretrain_path = config.pretrain_model_path)
    return prediction

if __name__ == '__main__':
    ##############intra sentence level########################
    data_path = config.intra_path + config.clean_data
    save_path = config.result_path + config.intra
    #####train and predict########
    intra_prediction = train_and_test(data_path = data_path,
                                      save_path = save_path,
                                      is_intra = True,
                                      func = KR_ERE)

    if config.is_document:
        ##############inter sentence level########################
        data_path = config.inter_path + config.clean_data
        ctd_path = config.result_path + config.inter
        #####train and predict########
        inter_prediction = train_and_test(data_path = data_path,
                                          save_path = save_path,
                                          is_intra = False,
                                          func = KR_ERE)
        merge_intra_inter(intra_path = intra_prediction,
                          inter_path = inter_prediction,
                          save_path = config.result_path + config.document)

