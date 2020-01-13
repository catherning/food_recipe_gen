import csv
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
import recipe_1m_analysis.ingr_normalization as ingr_norm
from recipe_1m_analysis.utils import Vocabulary
from recipe_gen.seq2seq_utils import DATA_FILES, FOLDER_PATH



def getKnowPairs(filepath):
    with open(os.path.join(known_file), 'rb') as f:
        known_pairs = pd.read_csv(f)

    known_dict = {}
    count_e = 0
    for row_idx, row in known_pairs.iterrows():
        try:
            ingr1_n = ingr_norm.normalize_ingredient(row["ingr1"])
            ingr2_n = ingr_norm.normalize_ingredient(row["ingr2"])
            known_dict[frozenset((ingr1_n.name, ingr2_n.name))] = row["npmi"]
        except AttributeError:
            count_e += 1

    print("Removed {} false pairings".format(count_e))
    return known_dict


def getMainIngr(vocab_ingrs):
    # Select the main ingredient from the clustering of ingredients. Otherwise, too many combinations
    vocab_main_ingr = Vocabulary()
    for k, v in vocab_ingrs.idx2word.items():
        main_ingr = None
        min_len = 100
        for ingr in v:
            if len(ingr.split("_")) < min_len:
                main_ingr = ingr
                min_len = len(ingr.split("_"))
        try:
            vocab_main_ingr.add_word(
                ingr_norm.normalize_ingredient(main_ingr).name, k)
        except:
            print(main_ingr)

    # TODO: remove special tokens sos, eos, pad, unk more properly (doesn't work for now)
    del vocab_main_ingr.word2idx["<"]
    del vocab_main_ingr.idx2word[0]

    list_ingr = list(vocab_main_ingr.word2idx.keys())

    return list_ingr


def generatePairings(list_ingr, save_path,known_dict):
    with open(save_path, mode='w', newline='') as f:
        csvwriter = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["ingr1", "ingr2"])
        for i, ingr1 in enumerate(list_ingr):
            for ingr2 in list_ingr[i+1:]:
                if frozenset((ingr1, ingr2)) not in known_dict:
                    csvwriter.writerow([ingr1, ingr2])


if __name__ == "__main__":
    path = os.path.join(
        os.getcwd(), "KitcheNette-master", "data")
    known_file =  os.path.join(path,"kitchenette_pairing_scores.csv")
    known_dict = getKnowPairs(known_file)

    with open(os.path.join(FOLDER_PATH, DATA_FILES[3]), 'rb') as f:
        vocab_ingrs = pickle.load(f)

    list_ingr = getMainIngr(vocab_ingrs)
    generatePairings(list_ingr, os.path.join(path,'main_pairing_prediction.csv'),known_dict)
