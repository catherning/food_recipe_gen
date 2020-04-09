#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.insert(0, os.getcwd())
import pickle
import itertools
import logging
import argparse
from recipe_gen.seq2seq_utils import RecipesDataset, FOLDER_PATH,DATA_FILES
from recipe_gen.main import argparser,PAIRING_PATH, init_logging, str2bool
# from nlgeval import compute_individual_metrics,compute_metrics
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score


# argparser = argparse.ArgumentParser()
# argparser.register('type', 'bool', str2bool)

argparser.add_argument('--eval-folder', type=str,
                       help='Generated recipes path (only the data)')



def processOutput(folder_path,ref,gen_ref=False):
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

def main(args,LOGGER):


    with open(os.path.join(args.data_folder,args.test_file),'rb') as f:
        ref=pickle.load(f)
        
    EVAL_FOLDER = os.path.join(os.getcwd(),"recipe_gen","results",args.model_name,args.eval_folder)

    processed=processOutput(EVAL_FOLDER,ref,True)

    bleu={i:0 for i in range(1,5)}
    bleu_w = {1:(1, 0, 0, 0),
            2:(0.5, 0.5, 0, 0),
            3:(0.33, 0.33, 0.33, 0),
            4:(0.25, 0.25, 0.25, 0.25)}
    
    meteor=0
    rouge=0
    #TODO: calc ROUGE
    t_ingr = 0
    r_ingr = 0
    added_ingr = 0
    for data in processed.values():
        bleu += sentence_bleu(data["ref"],data["gen"],weights=(0.5,0.5)) # Do more bleu scores
        for k,v in bleu.items():
            v+= sentence_bleu(data["ref"], data["gen"], weights=bleu_w[k])
            
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

    LOGGER.info("BLEU = {}".format(bleu/len(processed)))
    LOGGER.info("METEOR = {}".format(meteor/len(processed)))
    LOGGER.info("NB_INGR_INPUT = {}".format(sum(len(data["ingr"]) for data in processed.values())/len(processed)))
    LOGGER.info("INGR_MENTIONED_TARGET = {}".format(r_ingr/len(processed)))
    LOGGER.info("INGR_MENTIONED_GEN = {}".format(r_ingr/len(processed)))
    LOGGER.info("ADD_INGR_MENTIONED_GEN = {}".format(added_ingr/len(processed)))

if __name__ == "__main__":
    LOGGER = logging.getLogger()
    args = argparser.parse_args()
    init_logging(args)
    main(args,LOGGER)