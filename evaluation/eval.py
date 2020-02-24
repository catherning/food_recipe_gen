#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.insert(0, os.getcwd())
import pickle
import itertools
from recipe_gen.seq2seq_utils import RecipesDataset, FOLDER_PATH,DATA_FILES
from recipe_gen.main import args,PAIRING_PATH 
# from nlgeval import compute_individual_metrics,compute_metrics
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score

EVAL_FOLDER = os.path.join(os.getcwd(),"recipe_gen","results","Seq2seqIngrPairingAtt","02-21-18-49")

with open(os.path.join(args.data_folder,args.test_file),'rb') as f:
    ref=pickle.load(f)


def processOutput(folder_path,gen_ref=False):
    processed = {}
    # XXX: try to optimize code ?
    if gen_ref:
        with open(os.path.join(folder_path,"log.txt")) as f:
            for line in f:
                output = line.split("[")[1].split()
                if len(output[0])!=10 or output[0]=="train_mode": # improve, an arg name could have len 10 like the ex id
                    continue
                # TODO: check if punction with/without space is oks     
                # TODO: opt => direct can calculate the metrics instead of looping again
                processed[output[0]]={"ref":list(itertools.chain.from_iterable(ref[output[0]]["tokenized"])),
                                    "gen":output[1:-1],
                                    "ingr":[ingr.name for ingr in ref[output[0]]["ingredients"]]}
    return processed
  

processed=processOutput(EVAL_FOLDER,True)

bleu=0
meteor=0
t_ingr = 0
r_ingr = 0
added_ingr = 0
for data in processed.values():
    bleu += sentence_bleu(data["ref"],data["gen"],weights=(0.5,0.5)) # Do more bleu scores
    meteor += single_meteor_score(" ".join(data["ref"])," ".join(data["gen"]))

    for ingr in data["ingr"]:
        #ingr from the input that are mentioned
        if ingr in data["gen"]:
            t_ingr+=1
        else:
            # ingr added not from input
            added_ingr+=1
        if ingr in data["ref"]:
            r_ingr+=1

#TODO: get average ingr compatibility, average ingr compat for added ingr
        
    
print(bleu/len(processed))
print(meteor/len(processed))
print(sum(len(data["ingr"]) for data in processed.values())/len(processed))
print(r_ingr/len(processed))
print(t_ingr/len(processed))
