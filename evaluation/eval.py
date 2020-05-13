#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import itertools
import pickle
import sys
import os
import torch

from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu

sys.path.insert(0, os.getcwd())

from recipe_gen.seq2seq_utils import RecipesDataset, FOLDER_PATH, DATA_FILES
from recipe_gen.main import argparser, PAIRING_PATH, init_logging, str2bool, main as main_gen, getDefaultArgs
from recipe_gen.pairing_utils import PairingData


# from nlgeval import compute_individual_metrics,compute_metrics

# args.model_name
argparser.add_argument('--eval-folder', type=str,
                       help='Generated recipes path (only the data)')


def paramLogging(args):
    for k, v in args.defaults.items():
        try:
            if v is None or getattr(args, k) != v:
                args.logger.info("{} = {}".format(
                    k, getattr(args, k)))
        except AttributeError:
            continue


def processOutput(args):

    with open(os.path.join(args.data_folder, args.test_file), 'rb') as f:
        ref = pickle.load(f)

    EVAL_FOLDER = os.path.join(
        os.getcwd(), "recipe_gen", "results", args.model_name, args.eval_folder)

    processed = {}
    # XXX: try to optimize code ?
    try:
        with open(os.path.join(EVAL_FOLDER, "recipes_eval.pkl"), 'rb') as f:
            processed = pickle.load(f)

    except FileNotFoundError:
        with open(os.path.join(EVAL_FOLDER, "log.txt")) as f:
            for line in f:
                output = line.split("[")[1].split()
                # improve, an arg name could have len 10 like the ex id
                if output[0] in args.__dict__ or output[0]=="Dataset":
                    continue
                # TODO: check if punction with/without space is oks
                # TODO: opt => direct can calculate the metrics instead of looping again
                try:
                    processed[output[0]] = {"ref": list(itertools.chain.from_iterable(ref[output[0]]["tokenized"])),
                                            "gen": output[1:-1],
                                            "ingr": [ingr.name for ingr in ref[output[0]]["ingredients"]],
                                            "ref_step":len(ref[output[0]]["instructions"])}
                except KeyError:
                    processed["loss"]= output[3]
                    break

        with open(os.path.join(EVAL_FOLDER, "recipes_eval.pkl"), 'wb') as f:
            pickle.dump(processed, f)

    return processed

def loadPairing():   
    try:
        with open(os.path.join(os.path.dirname(args.pairing_path),"full_pairing.pkl"), 'rb') as f:
            pairing = pickle.load(f)
        print("Pairings loaded")
    except FileNotFoundError:
        FOLDER_PATH = "F:\\user\\Google Drive\\Catherning Folder\\THU\\Thesis\\Work\\Recipe datasets\\cuisine_classification"
        FILES = ["cleaned_data.pkl","full_data.pkl"]

        known_path = os.path.join(os.getcwd(),"KitcheNette_master","data","kitchenette_pairing_scores.csv")
        unk_path = os.path.join(os.getcwd(),"KitcheNette_master","results","prediction_unknowns_kitchenette_pretrained.mdl.csv")
        pairing_pickle = os.path.join(os.getcwd(),"KitcheNette_master","results","full_pairings.pkl")
        pairing = PairingData([unk_path,known_path], pickle_file=pairing_pickle, min_score=-1,trim=False)
    return pairing

def recipeScore(ingr_list,pairing):
    # TODO: get average ingr compatibility, average ingr compat for added ingr
    nb = 0
    score = 0
    for i,ingr1 in enumerate(ingr_list):
        for ingr2 in ingr_list[i+1:]:
            try:
                score += pairing.pairing_scores[pairing.pairedIngr[ingr1]][pairing.pairedIngr[ingr2]]
                nb+=1
                
            except KeyError:
                pass
    try:
        return score/nb
    except ZeroDivisionError:
        return 0  
    

def runEval(args):

    args_ = main_gen()
    args.eval_folder = os.path.split(args_.saving_path)[-1]


def main(args):
    LOGGER = args.logger
    paramLogging(args)
    if args.eval_folder is None:
        runEval(args)
        
    with open(os.path.join(FOLDER_PATH, DATA_FILES[3]), 'rb') as f:
        vocab_ingrs = pickle.load(f)

    processed = processOutput(args)
    NB_RECIPES = len(processed)-1
    print("{} recipes".format(NB_RECIPES))
    
    LOGGER.info("loss = {}".format(processed["loss"]))
    del processed["loss"]
    
    pairing = loadPairing()

    bleu = {i: 0 for i in range(1, 5)}
    bleu_w = {1: (1, 0, 0, 0),
              2: (0.5, 0.5, 0, 0),
              3: (0.33, 0.33, 0.33, 0),
              4: (0.25, 0.25, 0.25, 0.25)}

    meteor = 0
    rouge = 0
    # TODO: calc ROUGE
    target_ingr = 0
    gen_ingr = 0
    added_ingr = 0
    gen_step = 0
    target_step = 0
    gen_score = 0
    gen_mentioned_score = 0
    target_score = 0
    for data in processed.values():
        for k, v in bleu.items():
            v += sentence_bleu(data["ref"], data["gen"], weights=bleu_w[k])

        meteor += single_meteor_score(
            " ".join(data["ref"]), " ".join(data["gen"]))

        for ingr in data["ingr"]:
            # ingr from the input that are mentioned
            if ingr in data["ref"]:
                target_ingr += 1
        
        
        mentioned = set()
        for w in data["gen"]:
            if w in vocab_ingrs.word2idx and w not in mentioned:
                mentioned.add(w)
                if w in data["ingr"]:
                    gen_ingr += 1
                else:
                    added_ingr += 1
                    
        # Recipe score using KitcheNette pairings
        gen_score += recipeScore(list(mentioned)+data["ingr"],pairing)
        gen_mentioned_score += recipeScore(list(mentioned),pairing)
        target_score += recipeScore(data["ingr"],pairing)
                    
        target_step += data["ref_step"]
        gen_step += data["gen"].count("<eos>")

    # TODO: analysis by cuisine
    
    # Average for all recipes
    try:
        for k,v in bleu.items():
            LOGGER.info("BLEU{} = {}".format(k,v/NB_RECIPES))
                
        LOGGER.info("METEOR = {}".format(meteor/NB_RECIPES))
        LOGGER.info("NB_INGR_INPUT = {}".format(
            sum(len(data["ingr"]) for data in processed.values())/NB_RECIPES))
        LOGGER.info("INGR_MENTIONED_TARGET = {}".format(target_ingr/NB_RECIPES))
        LOGGER.info("INGR_MENTIONED_GEN = {}".format(gen_ingr/NB_RECIPES))
        LOGGER.info("ADD_INGR_MENTIONED_GEN = {}".format(
            added_ingr/NB_RECIPES))
        
        LOGGER.info("STEP_TARGET = {}".format(target_step/NB_RECIPES))
        LOGGER.info("STEP_GEN = {}".format(gen_step/NB_RECIPES))
        LOGGER.info("GEN_SCORE = {}".format(gen_score/NB_RECIPES))
        LOGGER.info("TARGET_SCORE = {}".format(target_score/NB_RECIPES))
    except ZeroDivisionError:
        print("There's no recipes")


if __name__ == "__main__":
    args = getDefaultArgs(argparser)
    args.logger = logging.getLogger()
    init_logging(args)
    main(args)
