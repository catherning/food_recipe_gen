#!/usr/bin/env python
# coding: utf-8

# https://github.com/facebookresearch/inversecooking/blob/master/src/build_vocab.py


import nltk
import pickle
import argparse
from collections import Counter
import json
import os
from tqdm import *
import numpy as np
import re
import sys
sys.path.insert(0, os.getcwd())

from recipe_1m_analysis.utils import Vocabulary
import recipe_1m_analysis.ingr_normalization as ingr_norm

# TODO: clean all this mess

def get_instruction(instruction, replace_dict, instruction_mode=True):
    instruction = instruction.lower()

    for rep, char_list in replace_dict.items():
        for c_ in char_list:
            if c_ in instruction:
                instruction = instruction.replace(c_, rep)
        instruction = instruction.strip()
    # remove sentences starting with "1.", "2.", ... from the targets
    if len(instruction) > 0 and instruction[0].isdigit() and instruction_mode:
        instruction = ''
    return instruction


def remove_plurals(counter_ingrs, ingr_clusters):
    del_ingrs = []

    for k, v in counter_ingrs.items():

        if len(k) == 0:
            del_ingrs.append(k)
            continue

        gotit = 0
        if k[-2:] == 'es':
            if k[:-2] in counter_ingrs.keys():
                counter_ingrs[k[:-2]] += v
                ingr_clusters[k[:-2]].extend(ingr_clusters[k])
                del_ingrs.append(k)
                gotit = 1

        if k[-1] == 's' and gotit == 0:
            if k[:-1] in counter_ingrs.keys():
                counter_ingrs[k[:-1]] += v
                ingr_clusters[k[:-1]].extend(ingr_clusters[k])
                del_ingrs.append(k)

    for item in del_ingrs:
        del counter_ingrs[item]
        del ingr_clusters[item]
    return counter_ingrs, ingr_clusters


def cluster_ingredients(counter_ingrs):
    new_counter = dict()
    new_ingr_cluster = dict()

    for k, v in counter_ingrs.items():
        k_split = k.split('_')
        w1 = k_split[-1]
        w2 = k_split[0]
        lw = [w1, w2]
        if len(k_split) > 1:
            w3 = k_split[0] + '_' + k_split[1]
            w4 = k_split[-2] + '_' + k_split[-1]

            lw = [w1, w2, w4, w3]

        gotit = 0
        for w in lw:
            if w in counter_ingrs.keys():
                # check if its parts are
                parts = w.split('_')
                if len(parts) > 0:
                    if parts[0] in counter_ingrs.keys():
                        w = parts[0]
                    elif parts[1] in counter_ingrs.keys():
                        w = parts[1]
                if w in new_counter.keys():
                    new_counter[w] += v
                    new_ingr_cluster[w].append(k)
                else:
                    new_counter[w] = v
                    new_ingr_cluster[w] = [k]
                gotit = 1
                break
        if gotit == 0:
            new_counter[k] = v
            new_ingr_cluster[k] = [k]

    return new_counter, new_ingr_cluster


def update_counter(list_, counter_toks, istrain=False):
    for sentence in list_:
        tokens = nltk.tokenize.word_tokenize(sentence)
        if istrain:
            counter_toks.update(tokens)


def raw_instr(instrs, instrs_list, replace_dict_instrs):
    acc_len = 0
    for instr in instrs:
        instr = instr['text']
        instr = get_instruction(instr, replace_dict_instrs)
        if len(instr) > 0:
            instrs_list.append(instr)
            acc_len += len(instr)
    return acc_len

def genTokVoc(counter_toks):
    ## Recipe vocab
    # Create a vocab wrapper and add some special tokens.
    vocab_toks = Vocabulary()
    vocab_toks.add_word('<pad>')
    vocab_toks.add_word('<sos>')
    vocab_toks.add_word('<eos>')
    vocab_toks.add_word('<unk>')

    # Add the words to the vocabulary.
    for word, cnt in counter_toks.items():
        # If the word frequency is less than 'threshold', then the word is discarded.
        if cnt >= args.threshold_words:
            vocab_toks.add_word(word)

    with open(os.path.join(args.save_path, args.suff+'recipe1m_vocab_toks.pkl'), 'wb') as f:
        pickle.dump(vocab_toks, f)
    print("Total token vocabulary size: {}".format(len(vocab_toks)))

def cleanCounterIngr(counter_ingrs):
    # manually add missing entries for better clustering
    base_words = ['peppers', 'tomato', 'spinach_leaves', 'turkey_breast', 'lettuce_leaf',
                'chicken_thighs', 'milk_powder', 'bread_crumbs', 'onion_flakes',
                'red_pepper', 'pepper_flakes', 'juice_concentrate', 'cracker_crumbs', 'hot_chili',
                'seasoning_mix', 'dill_weed', 'pepper_sauce', 'sprouts', 'cooking_spray', 'cheese_blend',
                'basil_leaves', 'pineapple_chunks', 'marshmallow', 'chile_powder',
                'cheese_blend', 'corn_kernels', 'tomato_sauce', 'chickens', 'cracker_crust',
                'lemonade_concentrate', 'red_chili', 'mushroom_caps', 'mushroom_cap', 'breaded_chicken',
                'frozen_pineapple', 'pineapple_chunks', 'seasoning_mix', 'seaweed', 'onion_flakes',
                'bouillon_granules', 'lettuce_leaf', 'stuffing_mix', 'parsley_flakes', 'chicken_breast',
                'basil_leaves', 'baguettes', 'green_tea', 'peanut_butter', 'green_onion', 'fresh_cilantro',
                'breaded_chicken', 'hot_pepper', 'dried_lavender', 'white_chocolate',
                'dill_weed', 'cake_mix', 'cheese_spread', 'turkey_breast', 'chicken_thighs', 'basil_leaves',
                'mandarin_orange', 'laurel', 'cabbage_head', 'pistachio', 'cheese_dip',
                'thyme_leave', 'boneless_pork', 'red_pepper', 'onion_dip', 'skinless_chicken', 'dark_chocolate',
                'canned_corn', 'muffin', 'cracker_crust', 'bread_crumbs', 'frozen_broccoli',
                'philadelphia', 'cracker_crust', 'chicken_breast']

    for base_word in base_words:
        if base_word not in counter_ingrs.keys():
            counter_ingrs[base_word] = 1

    counter_ingrs, cluster_ingrs = cluster_ingredients(counter_ingrs)
    counter_ingrs, cluster_ingrs = remove_plurals(counter_ingrs, cluster_ingrs)

    return counter_ingrs, cluster_ingrs

def genIngrVoc(counter_ingrs):
    counter_ingrs, cluster_ingrs = cleanCounterIngr(counter_ingrs)

    ## Ingredient vocab
    # Create a vocab wrapper for ingredients
    vocab_ingrs = Vocabulary()
    vocab_ingrs.add_word('<pad>')
    vocab_ingrs.add_word('<sos>')
    vocab_ingrs.add_word('<eos>')
    idx = vocab_ingrs.add_word('<unk>')

    # Add the ingredients to the vocabulary.
    for word,cnt in counter_ingrs.items():
        if cnt >= args.threshold_ingrs:
            for ingr in cluster_ingrs[word]:
                idx = vocab_ingrs.add_word(ingr, idx)
            idx += 1

    with open(os.path.join(args.save_path, args.suff+'recipe1m_vocab_ingrs.pkl'), 'wb') as f:
        pickle.dump(vocab_ingrs, f)
    print("Total ingr vocabulary size: {}".format(len(vocab_ingrs)))

    return vocab_ingrs, counter_ingrs

def clean_count(args, dets, idx2ind, layer1, replace_dict_ingrs, replace_dict_instrs):
    #####
    # 1. Count words in dataset and clean
    #####

    ingrs_file = os.path.join(args.save_path, 'allingrs_count.pkl')
    instrs_file = os.path.join(args.save_path, 'allwords_count.pkl')

    if os.path.exists(ingrs_file) and os.path.exists(instrs_file) and not args.forcegen:
        print("loading pre-extracted word counters")
        with open(ingrs_file, 'rb') as f:
            counter_ingrs = pickle.load(f)
        with open(instrs_file, 'rb') as f:
            counter_toks = pickle.load(f)

    else:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        counter_toks = Counter()
        counter_ingrs = Counter()

        for i, entry in tqdm(enumerate(layer1)):

            # get all instructions for this recipe
            instrs = entry['instructions']

            instrs_list = []
            ingrs_list = []

            # retrieve pre-detected ingredients for this entry
            det_ingrs = dets[idx2ind[entry['id']]]['ingredients']

            valid = dets[idx2ind[entry['id']]]['valid']

            for j, det_ingr in enumerate(det_ingrs):
                if len(det_ingr) > 0 and valid[j]:
    
                    det_ingr_undrs = ingr_norm.normalize_ingredient(det_ingr["text"])

                    if det_ingr_undrs is not None:
                        ingrs_list.append(det_ingr_undrs)

            # get raw text for instructions of this entry
            acc_len = raw_instr(instrs, instrs_list, replace_dict_instrs)

            # discard recipes with too few or too many ingredients or instruction words
            if len(ingrs_list) < args.minnumingrs \
                    or len(instrs_list) < args.minnuminstrs \
                    or len(instrs_list) >= args.maxnuminstrs \
                    or len(ingrs_list) >= args.maxnumingrs \
                    or acc_len < args.minnumwords:
                continue

            # tokenize sentences and update counter
            update_counter(instrs_list, counter_toks,
                           istrain=entry['partition'] == 'train')
            title = nltk.tokenize.word_tokenize(entry['title'].lower())
            if entry['partition'] == 'train':
                counter_toks.update(title)
                ingr_str = [ingr.name for ingr in ingrs_list]
                counter_ingrs.update(ingr_str)
                counter_toks.update(ingr_str) # add ingr to whole recipe vocab

        genTokVoc(counter_toks)
        vocab_ingrs, counter_ingrs = genIngrVoc(counter_ingrs)

        with open(ingrs_file, 'wb') as f:
            pickle.dump(counter_ingrs, f)

        with open(instrs_file, 'wb') as f:
            pickle.dump(counter_toks, f)

    return vocab_ingrs


def tokenize_dataset(args, dets, idx2ind, layer1, replace_dict_ingrs, replace_dict_instrs, vocab_ingrs):
    ######
    # 2. Tokenize and build dataset based on vocabularies.
    ######

    dataset = {'train': {}, 'val': {}, 'test': {}}

    for i, entry in tqdm(enumerate(layer1)):

        # get all instructions for this recipe
        instrs = entry['instructions']

        instrs_list = []
        ingrs_list = []

        # retrieve pre-detected ingredients for this entry
        det_ingrs = dets[idx2ind[entry['id']]]['ingredients']
        valid = dets[idx2ind[entry['id']]]['valid']
        labels = []

        for j, det_ingr in enumerate(det_ingrs):
            if len(det_ingr) > 0 and valid[j]:
                det_ingr_undrs = ingr_norm.normalize_ingredient(det_ingr["text"])

                if det_ingr_undrs is not None:
                    ingrs_list.append(det_ingr_undrs)
                    label_idx = vocab_ingrs(det_ingr_undrs.name)
                    if label_idx is not vocab_ingrs('<pad>') and label_idx not in labels:
                        labels.append(label_idx)

        # get raw text for instructions of this entry
        acc_len = raw_instr(instrs, instrs_list, replace_dict_instrs)

        # we discard recipes with too many or too few ingredients or instruction words
        if len(labels) < args.minnumingrs \
                or len(instrs_list) < args.minnuminstrs \
                or len(instrs_list) > args.maxnuminstrs \
                or len(labels) > args.maxnumingrs \
                or acc_len < args.minnumwords:
            continue

        # tokenize sentences
        toks = []

        for instr in instrs_list:
            tokens = nltk.tokenize.word_tokenize(instr)
            toks.append(tokens)

        title = nltk.tokenize.word_tokenize(entry['title'].lower())

        newentry = {'instructions': instrs_list, 'tokenized': toks,
                    'ingredients': ingrs_list, 'title': title}
        dataset[entry['partition']][entry['id']]=newentry

    print("Dataset size:")
    for split in dataset.keys():
        with open(os.path.join(args.save_path, args.suff+'recipe1m_' + split + '.pkl'), 'wb') as f:
            pickle.dump(dataset[split], f)

        print(split, ':', len(dataset[split]))


def main(args):
    """
    Builds vocab recipe1m
    :param args:
    :return:
    """

    print("Loading data...")
    with open(os.path.join(args.recipe1m_path, 'det_ingrs.json'), 'r') as f:
        dets = json.load(f)

    with open(os.path.join(args.recipe1m_path, 'layer1.json'), 'r') as f:
        layer1 = json.load(f)

    print("Loaded data.")
    print("Found %d recipes in the dataset." % (len(layer1)))
    replace_dict_ingrs = {'and': ['&', "'n"], '': [
        '%', ',', '.', '#', '[', ']', '!', '?']}
    replace_dict_instrs = {'and': ['&', "'n"], '': ['#', '[', ']','.']} #Added dot because keep gen dot.

    idx2ind = {}
    for i, entry in enumerate(dets):
        idx2ind[entry['id']] = i

    #####
    # 1. Count words in dataset and clean
    #####
    if args.forcegen_all:
        vocab_ingrs = clean_count(
            args, dets, idx2ind, layer1, replace_dict_ingrs, replace_dict_instrs)
    else:
        with open(os.path.join(args.save_path, args.suff+'recipe1m_vocab_ingrs.pkl'), 'rb') as f:
            vocab_ingrs = pickle.load(f)
        print("Vocab ingrs loaded.")

    ######
    # 2. Tokenize and build dataset based on vocabularies.
    ######
    tokenize_dataset(args, dets, idx2ind, layer1,
                     replace_dict_ingrs, replace_dict_instrs, vocab_ingrs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe1m_path', type=str,
                        default='path/to/recipe1m',
                        help='recipe1m path')

    parser.add_argument('--save_path', type=str, default=os.path.join(os.getcwd(), "recipe_1m_analysis", 'data'),
                        help='path for saving vocabulary wrapper')

    parser.add_argument('--suff', type=str, default='')

    parser.add_argument('--threshold_ingrs', type=int, default=10,
                        help='minimum ingr count threshold')

    parser.add_argument('--threshold_words', type=int, default=10,
                        help='minimum word count threshold')

    parser.add_argument('--maxnuminstrs', type=int, default=20,
                        help='max number of instructions (sentences)')

    parser.add_argument('--minnuminstrs', type=int, default=2,
                        help='min number of instructions (sentences)')

    parser.add_argument('--maxnumingrs', type=int, default=20,
                        help='max number of ingredients')

    parser.add_argument('--minnumingrs', type=int, default=4,
                        help='min number of ingredients')

    parser.add_argument('--minnumwords', type=int, default=20,
                        help='minimum number of characters in recipe')

    parser.add_argument('--forcegen', dest='forcegen', action='store_true')
    parser.add_argument('--forcegen-all', dest='forcegen_all', action='store_true')
    parser.set_defaults(forcegen=False)

    args = parser.parse_args()
    main(args)
