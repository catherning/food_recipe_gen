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
                processed[output[0]] = {"ref": list(itertools.chain.from_iterable(ref[output[0]]["tokenized"])),
                                        "gen": output[1:-1],
                                        "ingr": [ingr.name for ingr in ref[output[0]]["ingredients"]]}

        with open(os.path.join(EVAL_FOLDER, "recipes_eval.txt"), 'wb') as f:
            pickle.dump(processed, f)

    return processed


def runEval(args):

    args_ = main_gen()
    args.eval_folder = os.path.split(args_.save_folder)[-1]


def main(args, LOGGER):
    if args.eval_folder is None:
        runEval(args)

    processed = processOutput(args)

    bleu = {i: 0 for i in range(1, 5)}
    bleu_w = {1: (1, 0, 0, 0),
              2: (0.5, 0.5, 0, 0),
              3: (0.33, 0.33, 0.33, 0),
              4: (0.25, 0.25, 0.25, 0.25)}

    meteor = 0
    rouge = 0
    # TODO: calc ROUGE
    t_ingr = 0
    r_ingr = 0
    added_ingr = 0
    for data in processed.values():
        # Do more bleu scores
        bleu += sentence_bleu(data["ref"], data["gen"], weights=(0.5, 0.5))
        for k, v in bleu.items():
            v += sentence_bleu(data["ref"], data["gen"], weights=bleu_w[k])

        meteor += single_meteor_score(
            " ".join(data["ref"]), " ".join(data["gen"]))

        for ingr in data["ingr"]:
            # ingr from the input that are mentioned
            if ingr in data["gen"]:
                t_ingr += 1
            else:
                # ingr added not from input
                added_ingr += 1
            if ingr in data["ref"]:
                r_ingr += 1

    # TODO: get average ingr compatibility, average ingr compat for added ingr

    LOGGER.info("BLEU = {}".format(bleu/len(processed)))
    LOGGER.info("METEOR = {}".format(meteor/len(processed)))
    LOGGER.info("NB_INGR_INPUT = {}".format(
        sum(len(data["ingr"]) for data in processed.values())/len(processed)))
    LOGGER.info("INGR_MENTIONED_TARGET = {}".format(r_ingr/len(processed)))
    LOGGER.info("INGR_MENTIONED_GEN = {}".format(r_ingr/len(processed)))
    LOGGER.info("ADD_INGR_MENTIONED_GEN = {}".format(
        added_ingr/len(processed)))


if __name__ == "__main__":
    LOGGER = logging.getLogger()
    args = argparser.parse_args()
    init_logging(args)
    main(args, LOGGER)
