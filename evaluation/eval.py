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
from recipe_gen.main import argparser, PAIRING_PATH, init_logging, str2bool, main as main_gen

# from nlgeval import compute_individual_metrics,compute_metrics

# args.model_name
argparser.add_argument('--eval-folder', type=str,
                       help='Generated recipes path (only the data)')


def processOutput(args):

    with open(os.path.join(args.data_folder, args.test_file), 'rb') as f:
        ref = pickle.load(f)

    EVAL_FOLDER = os.path.join(
        os.getcwd(), "recipe_gen", "results", args.model_name, args.eval_folder)

    processed = {}
    # XXX: try to optimize code ?
    if os.path.isfile(os.path.join(EVAL_FOLDER, "recipes_eval.txt")):
        with open(os.path.join(EVAL_FOLDER, "recipes_eval.txt"), 'rb') as f:
            processed = pickle.load(f)

    else:
        with open(os.path.join(EVAL_FOLDER, "log.txt")) as f:
            for line in f:
                output = line.split("[")[1].split()
                # improve, an arg name could have len 10 like the ex id
                if output[0] in args.__dict__:
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

        with open(os.path.join(EVAL_FOLDER, "recipes_eval.txt"), 'wb') as f:
            pickle.dump(processed, f)

    return processed


def runEval(args):

    args_ = main_gen()
    args.eval_folder = os.path.split(args_.save_folder)[-1]


def main(args, LOGGER):
    if args.eval_folder is None:
        runEval(args)
        
    with open(os.path.join(FOLDER_PATH, DATA_FILES[3]), 'rb') as f:
        vocab_ingrs = pickle.load(f)

    processed = processOutput(args)
    NB_RECIPES = len(processed)
    
    LOGGER.info("loss = {}".format(processed["loss"]))
    del processed["loss"]

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
    for data in processed.values():
        for k, v in bleu.items():
            v += sentence_bleu(data["ref"], data["gen"], weights=bleu_w[k])

        meteor += single_meteor_score(
            " ".join(data["ref"]), " ".join(data["gen"]))

        for ingr in data["ingr"]:
            # ingr from the input that are mentioned
            if ingr in data["ref"]:
                target_ingr += 1
                
        for w in data["gen"]:
            if w in vocab_ingrs.word2idx:
                if w in data["ingr"]:
                    gen_ingr += 1
                else:
                    added_ingr += 1
                    
        target_step += data["ref_step"]
        gen_step += data["gen"].count("<eos>")

    
    # Average for all recipes
    # TODO: get average ingr compatibility, average ingr compat for added ingr
    for k,v in bleu.items():
        try:
            LOGGER.info("BLEU{} = {}".format(k,v/NB_RECIPES))
        except ZeroDivisionError:
            LOGGER.info("BLEU{} = {}".format(k, 0))
            
    try:
        LOGGER.info("METEOR = {}".format(meteor/NB_RECIPES))
    except:
        LOGGER.info("METEOR = {}".format(0))
        
    LOGGER.info("NB_INGR_INPUT = {}".format(
        sum(len(data["ingr"]) for data in processed.values())/NB_RECIPES))
    LOGGER.info("INGR_MENTIONED_TARGET = {}".format(target_ingr/NB_RECIPES))
    LOGGER.info("INGR_MENTIONED_GEN = {}".format(gen_ingr/NB_RECIPES))
    LOGGER.info("ADD_INGR_MENTIONED_GEN = {}".format(
        added_ingr/NB_RECIPES))
    
    LOGGER.info("STEP_TARGET = {}".format(target_step/NB_RECIPES))
    LOGGER.info("STEP_GEN = {}".format(gen_step/NB_RECIPES))


if __name__ == "__main__":
    LOGGER = logging.getLogger()
    args = argparser.parse_args()
    init_logging(args)
    main(args, LOGGER)
