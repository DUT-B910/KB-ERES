# -*- coding: utf-8 -*-

import os
import subprocess

def evaluating (read_path,write_path):
    method = 'java -cp ./bc5cdr_eval.jar ncbi.bc5cdr_eval.Evaluate relation CID PubTator ' \
             './CDR_TestSet.PubTator.txt ' + read_path  + '>>' + write_path
    subprocess.call(method, shell=True)

def evaluate_score(model_name,path,is_intra):
    eval_path = './results/' +model_name + '/'
    intra = 'intra_eval_result.txt' if is_intra else 'inter_eval_result.txt'
    if os.path.isdir(eval_path):
        pass
    else:
        os.makedirs(eval_path)
    evaluating(path, eval_path + intra)

if __name__ == '__main__':
    path = './results/KCN/final_result.txt'
    save_path ='./results/KCN/final_eval.txt'
    evaluating(path,save_path)
