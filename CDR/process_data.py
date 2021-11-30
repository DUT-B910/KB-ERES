# -*- coding: utf-8 -*-

'''
Filename : process_data.py
Function : Add knowledge bases into entity pair instances
           Pickle all the training/development/test instances into pkl
'''

import pickle
import os
import numpy as np
from config import Config

np.random.seed(1337)
config = Config()
wordNum = 0
word_index = {}
entity_index = {}
relation_index ={}
entity_rel = {}#defaultdict(lambda:3)
entity_vecs = []
relation_vecs = []

def entity2ID(entity_path):
    #load the id of entities in knowledge base
    with open (entity_path,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            entities = line[0].strip().split('|')
            for entity in entities:
                entity_index[entity] = int(line[1])

def relation2ID(relation_path):
    #load the id of the relation in knowledge base
    with open (relation_path,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            relation_index[line[0]] = int(line[1])
        if 'NULL' not in relation_index:
            relation_index['NULL'] = len(relation_index)

def load_kgfile(kg_path):
    #load the chem-dis relation in knowledge base, the format is "C000xxx D000xxx NONE"
    with open (kg_path,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            entities_1 = line[0].strip().split('|')
            entities_2 = line[1].strip().split('|')
            for entity_1 in entities_1:
                for entity_2 in entities_2:
                    entity_rel[entity_1+entity_2] = relation_index[line[2]]

def load_entity (file_path):
    with open (file_path,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            line = np.array([float(num) for num in line])
            entity_vecs.append(line)

def load_relation (file_path):
    with open (file_path,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            line = np.array([float(num) for num in line])
            relation_vecs.append(line)
        none_rel = np.random.uniform(-0.25, 0.25, size= 100)
        relation_vecs.append(none_rel)

def get_triples (Entities):
    triples = []
    for entity in Entities:
        if entity[1] not in entity_index:
            entity_index[entity[1]] = len (entity_index)
            entity_vecs.append(np.random.uniform(-0.25,0.25,size=config.word_vec_dim))
        if entity[4] not in entity_index.keys():
            entity_index[entity[4]] = len (entity_index)
            entity_vecs.append(np.random.uniform(-0.25,0.25,size=config.word_vec_dim))
        if entity[1]+entity[4] not in entity_rel:
            entity_rel[entity[1]+entity[4]] = relation_index['NULL']
        triples.append([entity_index[entity[1]],entity_index[entity[4]], entity_rel[entity[1]+entity[4]]])
    return triples
        

def Sentence2ID(sentences):
    #to make the word in sentences into ID
    global wordNum
    sentencesID = []
    for sentence in sentences:
        sentenceid = []
        for word in sentence:
            if word not in word_index.keys():
                word_index[word] = wordNum
                wordNum += 1
            sentenceid.append(word_index[word])
        sentencesID.append(sentenceid)
    return sentencesID

def load_file(file_path):
    Label = []
    Entities = []
    Sentence = []
    with open(file_path,'r')as f:
        gold_num = int(f.readline())
        for line in f:
            line =line.strip().split('\t')
            Label.append(line[0]=='CID')
            entity = line[1].strip().split('_')
            #means articleID; ent1 ID; ent1's words start position;end position;ent2 ID; ent2 start position;end position
            #the sentence could be the sdp or sequence
            Entities.append ([entity[0],entity[1],int(entity[2]),int(entity[3]),entity[4],int(entity[5]),int(entity[6])])
            Sentence.append (line[2].lower().strip().split(' '))
    return Label, Entities, Sentence,  gold_num


def process_data(train_Label,train_Entities,train_Sentence,train_sum,\
                 dev_Label,dev_Entities,dev_Sentence,dev_sum,\
                 test_Label,test_Entities,test_Sentence,test_sum,\
                 output_path,data_path,word_vec_path = ''):
    max_sen_len = 0
    #MAX_SEQUENCE_LENGTH=10
    # calcuate the max length of doc, sentence and word
    for data in (train_Sentence, dev_Sentence, test_Sentence):
        for item in data:
            # print i
            max_sen_len = max(max_sen_len, len(item))

    print ('the max sentence length is : ', max_sen_len)
    train_SentenceID = Sentence2ID (train_Sentence)
    dev_SentenceID = Sentence2ID (dev_Sentence)
    test_SentenceID  = Sentence2ID (test_Sentence)
    print('Found %s unique tokens.' % wordNum)

    dict_writer =open(output_path+'/words.vocab','w')
    word_index_sorted = sorted(word_index.items(), key=lambda x: x[1])
    for k, v in word_index_sorted:
        dict_writer.write(k + ' ' + str(v) + '\n')

    dict_writer.close()

    # load word embeddings...
    embeddings_index = {}
    EMBEDDING_DIM = 100
    if word_vec_path:
        f = open(word_vec_path,'w')
        f.readline()
        for line in f:
            values = line.split()
            word = values[0].lower()
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors in Glove .' % len(embeddings_index))
    else:
        print('No pretrained word2vec, all embedding will be initialized randomly')

    embedding_matrix = np.random.uniform(-0.25, 0.25, size=(wordNum + 1, EMBEDDING_DIM))
    embedding_matrix[0] = np.zeros(shape=EMBEDDING_DIM, dtype='float32')
    number = 0
    vocab_oup =open(output_path+'/miss.vocab','w')
    for word in word_index.keys():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[word_index[word]] = embedding_vector
        else:
            number += 1
            vocab_oup.write(word+'\n')
    print ('total {} word not find in the word embeddings'.format(number))
    vocab_oup.close()

    # add the relation entities:
    train_Triples = get_triples (train_Entities)
    dev_Triples = get_triples (dev_Entities)
    test_Triples = get_triples (test_Entities)

    entity_matrix = np.asarray(entity_vecs,dtype = 'float32')
    relation_matirx = np.asarray(relation_vecs,dtype= 'float32')

    print ('dump the file ...')
    pickle.dump([train_SentenceID,train_Label,train_Entities,train_Triples,train_sum,\
                 dev_SentenceID,dev_Label,dev_Entities,dev_Triples,dev_sum,\
                 test_SentenceID,test_Label,test_Entities,test_Triples,test_sum,\
                 embedding_matrix,entity_matrix,relation_matirx], open(output_path+'/data.pkl', 'wb'))

if __name__ == '__main__':

    # intra-sentence level data
    train_data_path = config.intra_path + config.train_ins_path
    dev_data_path = config.intra_path + config.dev_ins_path
    test_data_path = config.intra_path + config.test_ins_path

    print("load entity index...")
    entity2ID (config.entity_index_path)

    print("load relation index...")
    relation2ID (config.relation_index_path)

    print("load kgfile...")
    load_kgfile (config.triple_path)

    print("load entity vectors in kb...")
    load_entity(config.entity_vec_path)

    print("load relation vectors in kb...")
    load_relation(config.relation_vec_path)

    print ('load intra instances')
    train_Label,train_Entities,train_Sentence,train_sum = load_file(train_data_path)
    dev_Label,dev_Entities,dev_Sentence,dev_sum = load_file(dev_data_path)
    test_Label,test_Entities,test_Sentence,test_sum = load_file(test_data_path)
    output_path = config.intra_path

    print ('dump intra instances')
    process_data(train_Label,train_Entities,train_Sentence,train_sum,\
                 dev_Label,dev_Entities,dev_Sentence,dev_sum,\
                 test_Label,test_Entities,test_Sentence,test_sum,\
                 output_path, config.test_ins_path,config.word_vec_path)
    
    #inter-sentence level
    train_data_path =  config.inter_path + config.train_ins_path
    dev_data_path =  config.inter_path + config.dev_ins_path
    test_data_path =  config.inter_path + config.test_ins_path

    print ('load inter instances')
    train_Label,train_Entities,train_Sentence,train_sum = load_file(train_data_path)
    dev_Label,dev_Entities,dev_Sentence,dev_sum = load_file(dev_data_path)
    test_Label,test_Entities,test_Sentence,test_sum = load_file(test_data_path)
    output_path = config.inter_path

    print ('dump inter instances')
    process_data(train_Label,train_Entities,train_Sentence,train_sum,\
                 dev_Label,dev_Entities,dev_Sentence,dev_sum,\
                 test_Label,test_Entities,test_Sentence,test_sum,\
                 output_path, config.test_ins_path,config.word_vec_path)
