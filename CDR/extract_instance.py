# -*- coding: utf-8 -*-
'''
Filename : extracct_instance.py
Function : From document level data into sentence level instance
           Extract instances from the given data, instances could be divided into 2 kinds:
           1. intra-sentence instances: entities in the same sentence, which we extract their co-existed sentence.
           2. inter-sentence instances: entities in different sentences, which we extract two sentences and connect them together.
'''

import pickle
import os
import numpy as np
from collections import defaultdict,OrderedDict
import re
import copy
from config import Config

ent_repl_dic = {}
Induced = []
Title = OrderedDict()

def norm_sentence(sentence):
    sentence = sentence + ' '
    sentence = sentence.replace("'s ",' ')
    sentence = sentence.replace("'",' ')
    sentence = sentence.replace('"',' ')

    sentence = sentence.replace(', ',' , ')
    sentence = sentence.replace(': ',' : ')
    sentence = sentence.replace('! ',' ! ')
    sentence = sentence.replace('? ',' ? ')
    sentence = sentence.replace('. ',' . ')
    sentence = sentence.replace('; ',' ; ')
    sentence = sentence.replace('-',' - ')
    
    sentence = re.sub('[\(\)\[\]\{\}]',' ',sentence)
    sentence = re.sub('\b[0-9]+\b','num',sentence)
    sentence = re.sub('\s+',' ',sentence)
    for _ in range (5):
        sentence = sentence.replace('  ',' ')

    return sentence.strip()

def generate_instance(article,entities):
    Intra_sent = []
    Inter_sent_raw = []
    del_overlap = {} #store the shortest length of the two entities, and only save the shortest instance.
    index_e1 = -1
    
    for ent_1 in entities:
        index_e1 += 1
        if ent_1[4] != 'Chemical':  #optional
            continue
        index_e2 = -1
        for ent_2 in entities:
            index_e2 += 1
            if ent_2[4] != 'Disease':  #optional
                continue
            if ent_1[1]>=ent_2[2]:
                first_ent, last_ent = ent_2, ent_1
            else:
                first_ent, last_ent = ent_1, ent_2
            sent_list = [i+1 for i in range(first_ent[2],last_ent[1]-1) if article[i:i+2] == '. ']
            len_e1_e2 = len ( article [first_ent[2]:last_ent[1]].strip().split(' ') ) #optinal, to judge and save the shortest length instance
            if len(sent_list) > 2:   #optional
                continue
            start = 0
            end = len(article)
            for i in range(first_ent[1],1,-1):
                if article[i-2:i]=='. ':
                    start = i
                    break
            for i in range(last_ent[2],len(article)-2):
                if article[i:i+2]=='. ':
                    end = i + 1
                    break
            if not len (sent_list):
                #intra sentence
                sentence_part1 = norm_sentence(article[start:first_ent[1]])
                first_ent_name = article[first_ent[1]:first_ent[2]]
                sentence_part2 = norm_sentence(article[first_ent[2]:last_ent[1]])
                last_ent_name = article[last_ent[1]:last_ent[2]]
                sentence_part3 = norm_sentence(article[last_ent[2]:end])
                first_ent_start = len(sentence_part1.split(' '))
                first_ent_end = first_ent_start + len(first_ent_name.split(' '))
                last_ent_start = len((' '.join([sentence_part1,first_ent_name,sentence_part2])).split(' '))
                last_ent_end = last_ent_start + len(last_ent_name.split(' '))
                sentence = ' '.join([sentence_part1,first_ent_name,sentence_part2,last_ent_name,sentence_part3])
                if first_ent[4] == 'Chemical':
                    pos_chem_start, pos_chem_end, pos_dis_start, pos_dis_end = first_ent_start, first_ent_end, last_ent_start, last_ent_end
                else:
                    pos_chem_start, pos_chem_end, pos_dis_start, pos_dis_end = last_ent_start, last_ent_end, first_ent_start, first_ent_end
                Intra_sent.append([ent_1[0], ent_1[5],str(pos_chem_start),str(pos_chem_end),ent_2[5],str(pos_dis_start),str(pos_dis_end),sentence])
                del_overlap[ent_1[5]+'_'+ent_2[5]] = 0
            else :
                #inter sentence
                sent1_part1 = norm_sentence(article[start:first_ent[1]])
                sent1_part2 = norm_sentence(article[first_ent[2]:sent_list[0]])
                sent2_part1 = norm_sentence(article[sent_list[-1]+1:last_ent[1]])
                sent2_part2 = norm_sentence(article[last_ent[2]:end])
                first_ent_name = article[first_ent[1]:first_ent[2]]
                last_ent_name = article[last_ent[1]:last_ent[2]]

                first_ent_start = len(sent1_part1.split(' '))
                first_ent_end = first_ent_start + len(first_ent_name.split(' '))
                last_ent_start = len(' '.join([sent1_part1,first_ent_name,sent1_part2,sent2_part1]))
                last_ent_end = last_ent_start + len (last_ent_name.split(' '))
                sentence = ' '.join([sent1_part1,first_ent_name,sent1_part2,sent2_part1,last_ent_name,sent2_part2])
                if first_ent[4] == 'Chemical':
                    pos_chem_start, pos_chem_end, pos_dis_start, pos_dis_end = first_ent_start, first_ent_end, last_ent_start, last_ent_end
                else:
                    pos_chem_start, pos_chem_end, pos_dis_start, pos_dis_end = last_ent_start, last_ent_end, first_ent_start, first_ent_end
                
                Inter_sent_raw.append([ent_1[0], ent_1[5],str(pos_chem_start),str(pos_chem_end),ent_2[5],str(pos_dis_start),str(pos_dis_end),sentence,len_e1_e2])

                if len_e1_e2 <= del_overlap.get(ent_1[5]+'_'+ent_2[5],len(article)):
                    del_overlap[ent_1[5]+'_'+ent_2[5]] = len_e1_e2
    Inter_sent = []
    for instance in Inter_sent_raw:
        if del_overlap[instance[1]+'_'+instance[4]] >= instance[-1]:
            Inter_sent.append(instance[:-1])
    return Intra_sent, Inter_sent

def load_original_data (data_path):
    Intra_Ins = []
    Inter_Ins = []
    label = defaultdict(lambda:'UN')
    sum = 0
    with open (data_path,'r') as f:
        line = f.readline()
        while not line == '':
            item = line.strip().split('|')
            Article = item[2]
            title_length = len(Article)
            title = Article
            line = f.readline()
            item = line.strip().split('|')
            Article = Article + ' ' + item[2]
            line = f.readline()
            entities = []
            while not line == '\n':
                item = line.strip().split('\t')
                if item[1] == 'CID':
                    label[item[0]+'_'+item[2]+'_'+item[3]] = 'CID'
                    sum += 1
                else:
                    if len(item) >= 6 and item[5][0] in ['C','D'] and item[5][1] != 'H':
                        #print (item)
                        item[1],item[2]=int(item[1]),int(item[2])
                        if not Article[max(item[1]-1,0)] =='(' or not Article[min(item[2],len(Article)-1)] == ')':
                            #statistic the inudced words.
                            if not item[2] == len(Article) and Article[item[2]] == '-':
                                induce = ''
                                for fig in Article[item[2]+1:]:
                                    if not fig.isalpha():
                                        break
                                    induce = induce + fig
                                if induce not in Induced:
                                    Induced.append(induce)
                            item[3] = item[3].lower()
                            item[5] = item[5].split('|')[0]
                            entities.append(item)
                line = f.readline()
            Title[item[0]] = title
            #here we add the filter of hyper.
            intra, inter = generate_instance(Article,entities)#,ex_word_dic)
            Intra_Ins.extend(intra)
            Inter_Ins.extend(inter)
            line = f.readline()

    return Intra_Ins,Inter_Ins,label,sum

def out_ins_data (ins, label, sum, ins_path):
    '''
    if not os.path.isdir(ins_path):
        os.makedirs(ins_path)
    '''
    ins_out = open(ins_path,'w')
    ins_out.write(str(sum) + '\n')
    for item in ins:
        ins_out.write (label[item[0]+'_'+item[1]+'_'+item[4]] + '\t' + '_'.join(item[:7]) + '\t' + item[-1] + '\n')
    ins_out.close()

def out_title (ID_path,title_path):
    out_ID = open(ID_path,'w')
    out_tit = open(title_path,'w')
    for key in Title.keys():
        out_ID.write(key +'\n')
        out_tit.write(Title[key] + '\n')
    out_ID.close()
    out_tit.close()

if __name__ == '__main__':

    # intra-sentence level data
    config = Config()
    CDR_train_data_path      = config.ori_train_path
    CDR_dev_data_path        = config.ori_dev_path
    CDR_test_data_path       = config.ori_test_path

    CDR_train_intra_ins_path      = config.intra_path + config.train_ins_path
    CDR_dev_intra_ins_path        = config.intra_path + config.dev_ins_path
    CDR_test_intra_ins_path       = config.intra_path + config.test_ins_path

    CDR_train_inter_ins_path      = config.inter_path + config.train_ins_path
    CDR_dev_inter_ins_path        = config.inter_path + config.dev_ins_path
    CDR_test_inter_ins_path       = config.inter_path + config.test_ins_path

    ID_path = 'title_ID.txt'
    title_path = 'title_seq.txt'

    print("load train data...")
    train_intra, train_inter, train_label,sum = load_original_data (CDR_train_data_path)
    print("output train data...")
    out_ins_data (ins = train_intra, 
                  label = train_label, 
                  sum = sum, 
                  ins_path = CDR_train_intra_ins_path)
    out_ins_data (ins = train_inter,
                  label = train_label, 
                  sum = sum, 
                  ins_path = CDR_train_inter_ins_path)

    print("load develop data...")
    dev_intra, dev_inter, dev_label,sum = load_original_data (CDR_dev_data_path)
    print("output dev data...")
    out_ins_data (ins = dev_intra,
                  label = dev_label, 
                  sum = sum, 
                  ins_path = CDR_dev_intra_ins_path)
    out_ins_data (ins = dev_inter, 
                  label = dev_label, 
                  sum = sum, 
                  ins_path = CDR_dev_inter_ins_path)

    print("load test data...")
    test_intra, test_inter, test_label,sum = load_original_data (CDR_test_data_path)
    print("output test data...")
    out_ins_data (ins = test_intra, 
                  label = test_label, 
                  sum = sum, 
                  ins_path = CDR_test_intra_ins_path)
    out_ins_data (ins = test_inter, 
                  label = test_label, 
                  sum = sum, 
                  ins_path = CDR_test_inter_ins_path)
    
    print ("output title...")
    out_title (ID_path,title_path)

    out_dic = open('./Induced.txt','w')
    for item in Induced:
        out_dic.write(item+'\n')
    out_dic.close()
