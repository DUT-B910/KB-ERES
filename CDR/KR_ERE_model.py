# -*- coding: utf-8 -*-
'''
Filename : KR_ERE_model.py
Function : Desigh the KR_ERE model used in our software
           train_fit: Train the model on training set
           test_eval: Test the model on development/test set
'''

import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from collections import defaultdict
import time
import argparse

from config import Config
from doc_level_evaluation import evaluate_score

mySeed = np.random.RandomState(1234)
class KR_ERE(nn.Module):
    def __init__(self, wordEmbed,entityEmbed,relationEmbed):
        super(KR_ERE,self).__init__()
        self.name = 'KR_ERE_model'
        self.wordEmbed = wordEmbed
        self.entityEmbed = entityEmbed
        self.relationEmbed = relationEmbed
        self.config = Config()

        self.batchSize = self.config.batch_size
        self.wordVectorLength = self.config.word_vec_dim

        self.vectorLength = self.config.word_vec_dim
        self.entityLength = self.config.kg_vec_dim
        self.classNumber = self.config.class_number
        self.numEpoches = self.config.epoch_number
        self.convdim = self.config.convolution_dim
        self.dropout = nn.Dropout(p=0.5)

        self.kernel_size = self.config.kernel_size

        self.convstc = nn.ModuleList([nn.Conv1d(self.vectorLength, self.convdim, K, padding=int((K-1)/2), bias = True) for K in self.kernel_size]).cuda()
        self.convsc = nn.ModuleList([nn.Conv1d(self.vectorLength, self.convdim, K, padding=int((K-1)/2), bias = True) for K in self.kernel_size]).cuda()
        self.convstd = nn.ModuleList([nn.Conv1d(self.vectorLength, self.convdim, K, padding=int((K-1)/2), bias = True) for K in self.kernel_size]).cuda()
        self.convsd = nn.ModuleList([nn.Conv1d(self.vectorLength, self.convdim, K, padding=int((K-1)/2), bias = True) for K in self.kernel_size]).cuda()
        
        self.chemical_W = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, self.entityLength ))).cuda (), requires_grad = True)
        self.chemical_b = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, 1 ))).cuda (), requires_grad = True)
        self.disease_W  = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, self.entityLength ))).cuda (), requires_grad = True)
        self.disease_b  = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, 1 ))).cuda (), requires_grad = True)

        self.LinearLayer_W = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, self.convdim * 2 * len(self.kernel_size) ))).cuda (), requires_grad = True)
        self.LinearLayer_b = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, 1 ))).cuda (), requires_grad = True)

        self.attention_W = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.entityLength, self.convdim ))).cuda (), requires_grad = True)
        self.attention_b = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.entityLength, 1))).cuda (), requires_grad = True)
        
        self.softmaxLayer_W = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, ( self.classNumber, self.convdim ))).cuda(), requires_grad=True)
        self.softmaxLayer_b = nn.Parameter(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, ( self.classNumber,1 ))).cuda(), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.loss_function = torch.nn.NLLLoss()  
  
    def forward(self, contxtWords, e1,e2, e1vs,e2vs,e1v,e2v, relation, senlength,is_train):
        softmaxLayer_W = self.softmaxLayer_W
        softmaxLayer_b = self.softmaxLayer_b
        vectorLength = self.vectorLength

        #generate entity expression

        E1 = torch.mm(self.chemical_W, e1.view(self.wordVectorLength,1)) + self.chemical_b
        E2 = torch.mm(self.disease_W, e2.view(self.wordVectorLength,1)) + self.disease_b
        
        contxt_chem = []
        contxt_dis = []
        gate_chem = []
        gate_dis = []
        for i,conv in enumerate(self.convstc):
            if i%2:
                contxt_chem.append(torch.tanh(conv(torch.cat([contxtWords.view(1,vectorLength,senlength),Variable(torch.zeros(1,vectorLength,1).cuda())],2))))
            else:
                contxt_chem.append(torch.tanh(conv(contxtWords.view(1,vectorLength,senlength))))

        for i,conv in enumerate(self.convstd):
            if i%2:
                contxt_dis.append(torch.tanh(conv(torch.cat([contxtWords.view(1,vectorLength,senlength),Variable(torch.zeros(1,vectorLength,1).cuda())],2))))
            else:
                contxt_dis.append(torch.tanh(conv(contxtWords.view(1,vectorLength,senlength))))

        for i,conv in enumerate(self.convsc):
            if i%2:
                gate_chem.append(torch.relu(conv(torch.cat([contxtWords.view(1,vectorLength,senlength),Variable(torch.zeros(1,vectorLength,1).cuda())],2))+ E1.view(1,self.convdim,1)))# 
            else:
                gate_chem.append(torch.relu(conv(contxtWords.view(1,vectorLength,senlength)) + E1.view(1,self.convdim,1)))#

        for i,conv in enumerate(self.convsd):
            if i%2:
                gate_dis.append(torch.relu(conv(torch.cat([contxtWords.view(1,vectorLength,senlength),Variable(torch.zeros(1,vectorLength,1).cuda())],2))+ E2.view(1,self.convdim,1)))# 
            else:
                gate_dis.append(torch.relu(conv(contxtWords.view(1,vectorLength,senlength)) + E2.view(1,self.convdim,1)))#
    
        contxtWords_chem = [(i*j).squeeze(0) for i, j in zip(contxt_chem, gate_chem)]
        contxtWords_dis = [(i*j).squeeze(0) for i, j in zip(contxt_dis, gate_dis)]

        contxtWords0_chem = []
        contxtWords0_dis = []
        for contxt_chem,contxt_dis in zip(contxtWords_chem,contxtWords_dis):
            att = self.softmax( torch.mm(relation.view(1,self.entityLength), torch.tanh(torch.mm(self.attention_W,contxt_chem) + self.attention_b)) )
            contxtWords0_chem.append(torch.mm(att,contxt_chem.transpose(0,1)).view(self.convdim,1))

            att = self.softmax( torch.mm(relation.view(1,self.entityLength), torch.tanh(torch.mm(self.attention_W,contxt_dis) + self.attention_b)) )
            contxtWords0_dis.append(torch.mm(att,contxt_dis.transpose(0,1)).view(self.convdim,1))
        contxtWords0_chem = torch.cat(contxtWords0_chem,0)
        contxtWords0_dis = torch.cat(contxtWords0_dis,0)
        contxtWords0 = torch.cat([contxtWords0_chem,contxtWords0_dis],0)

        linearLayerOut = torch.relu(torch.mm(self.LinearLayer_W,self.dropout(contxtWords0)) + self.LinearLayer_b)
        finallinearLayerOut = torch.mm(softmaxLayer_W,linearLayerOut) + softmaxLayer_b
        return finallinearLayerOut
    
    def load_pretrain_parameters (self, parameters_path = None):
        if parameters_path != None:
            checkpoint = torch.load(parameters_path)
            self.load_state_dict(checkpoint)

    def train_fit(self, trainset,trainLabel,train_gold_num,valset,valLabel,val_gold_num,testset,testLabel,test_gold_num,resultOutput, is_intra = True,pretrain_path = ''):
        F1 = 0
        indicates=list(range(len(trainset)))
        trainsetSize = len(trainset)

        if pretrain_path:
            self.load_pretrain_parameters(pretrain_path)

        if is_intra:
            learn_rate = self.config.intra_lr
        else:
            learn_rate = self.config.inter_lr

        optimizer = optim.Adam(self.parameters(), lr = learn_rate)

        for epoch_idx in range (self.numEpoches):
            mySeed.shuffle(indicates)
            total_loss = Variable(torch.FloatTensor([0]).cuda(), requires_grad=True)
            sum_loss= 0.0
            print("=====================================================================")
            print("epoch " + str(epoch_idx) + ", trainSize: " + str(trainsetSize))

            count = 0
            correct = 0
            tp = 0
            tp_fp = 0
            predict_dic = []
            time0 = time.time()
            self.train()
            for i in range(len(indicates)):
                sentid, sentwords,e1id,e2id,e1,e2,e1vs,e2vs,e1v,e2v,relation, senlength= trainset[indicates[i]]
                finallinearLayerOut =  self.forward(
                    sentwords,
                    e1,
                    e2,
                    e1vs,
                    e2vs,
                    e1v,
                    e2v,
                    relation,
                    senlength,
                    True
                )
                log_prob = F.log_softmax(finallinearLayerOut.view(1, self.classNumber),dim = 1)
                loss = self.loss_function(log_prob, Variable(torch.LongTensor([trainLabel[indicates[i]]]).cuda()))
                classification = self.softmax(finallinearLayerOut.view(1, self.classNumber))

                total_loss = torch.add(total_loss, loss)

                predict = np.argmax(classification.cpu().data.numpy())
                prediction = [sentid,e1id,e2id,predict]
                if predict == trainLabel[indicates[i]]:
                    correct += 1.0       
                count += 1
                if predict and prediction not in predict_dic:
                    predict_dic.append (prediction)
                    tp_fp += 1
                    if predict == trainLabel[indicates[i]]:
                        tp += 1
####################Update#######################
                if count % self.batchSize == 0:
                    total_loss = total_loss/self.batchSize
                    total_loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss = Variable(torch.FloatTensor([0]).cuda(),requires_grad = True)
            optimizer.step()
            optimizer.zero_grad()
            
            self.eval()
            resultStream = open(resultOutput + "vresult_" + str(epoch_idx) + ".txt", 'w')
            probPath   = resultOutput + "vprob_" + str(epoch_idx) + ".txt"
            VP,VR,VF = self.test_eval(valset,valLabel, val_gold_num,resultStream, probPath)

            resultStream = open(resultOutput + "tresult_" + str(epoch_idx) + ".txt", 'w')
            probPath   = resultOutput + "tprob_" + str(epoch_idx) + ".txt"
            TP,TR,TF = self.test_eval(testset,testLabel, test_gold_num, resultStream, probPath)
            resultStream.close()

            if VF >= F1:
                F1 = VF
                file_path = resultOutput + "tresult_" + str(epoch_idx) + ".txt"
                torch.save(self.state_dict(),self.config.model_save_path)
####################Update#######################
            train_P = tp/max(tp_fp,1)
            train_R = tp/train_gold_num
            train_F = 2*train_P*train_R/max(0.0001,(train_P+train_R))

            time1 = time.time()
            print (tp,tp_fp,train_gold_num)
            print("train P: ", train_P, " R: ", train_R , " F1: ", train_F)
            print("val   P: ", VP, " R: ", VR , " F1: ", VF)
            print("test  P: ", TP, " R: ", TR , " F1: ", TF)
            print("Iteration", epoch_idx, "Loss", total_loss.cpu().data.numpy()[0] / self.batchSize, "train Acc: ", float(correct / count) , "time: ", str(time1 - time0))
        return file_path

    def test_eval(self, testset, testLabel,gold_correct,resultStream, probPath):
        time0 = time.time()
        probs = []
        predict_dic = []
        correct = 0
        count = 0
        for i in range(len(testset)):
            sentid, sentwords, e1id,e2id,e1,e2,e1vs,e2vs,e1v,e2v,relation,senlength = testset[i]
            finallinearLayerOut =  self.forward(
                sentwords,
                e1,
                e2,
                e1vs,
                e2vs,
                e1v,
                e2v,
                relation,
                senlength,
                False
            )
            classification = self.softmax(finallinearLayerOut.view(1, self.classNumber))
            prob = classification.cpu().data.numpy().reshape(self.classNumber)
            predict = np.argmax(prob)
            probs.append(prob)
            prediction = [sentid,e1id,e2id,predict]
            if predict and (prediction not in predict_dic):
                resultStream.write("\t".join([sentid, "CID", e1id, e2id]) + "\n")
                predict_dic.append(prediction)
                if predict and testLabel[i]:
                    correct += 1
                count += 1

        P = correct/max(count,1)
        R = correct/gold_correct
        F = 2*P*R/max(0.0001,(P+R))
        if probPath:
            np.savetxt(probPath, probs, '%.5f',delimiter=' ')

        time1 = time.time()
        print("test time : ", str(time1 - time0))
        return P,R,F
