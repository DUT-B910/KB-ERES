# -*- coding: utf-8 -*-
'''
Filename : merge_result.py
Function : merge the intra-sentence result and inter-sentence result as final results
           Note that we choose the epoch where the development results are the best, to combine the results of development set and test set.
'''

import numpy as numpy
import os

instance = []
probability = []

def merge(result_path):
    with open (result_path,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if line not in instance:
                instance.append(line)
                probability.append('0.5')

def out_result(merge_path):
    out = open (merge_path,'w')
    for i in range(len(instance)):
        ins = '\t'.join(instance[i])
        out.write(ins+'\t'+probability[i]+'\n')
    out.close()

def merge_intra_inter(intra_path,inter_path,save_path):
    intra_epoch = intra_path.split('.')[0].split('_')[-1]
    inter_epoch = inter_path.split('.')[0].split('_')[-1]
    merge_path = 'tmerge_result_' + intra_epoch +'_'+ inter_epoch+'.txt'
    merge(intra_path)
    merge(inter_path)
    out_result(merge_path)
