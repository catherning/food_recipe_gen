àimport csv
import heapq
import os
import pickle
import sys

import pandas as pd
sys.path.insert(0, os.getcwd())

import recipe_1m_analysis.ingr_normalization as ingr_norm
from recipe_1m_analysis.utils import Vocabulary
from recipe_gen.seq2seq_utils import DATA_FILES, FOLDER_PATH


filepath = "D:\\Documents\\THU\\KitcheNette-master\\results\\prediction_unknowns_smaller_kitchenette_trained.mdl.csv"


class PairingData:
    def __init__(self, filepath, min_score=0, top_k=20):

        self.min_score = min_score
        self.top_k = top_k
        # XXX:repeat with other data classes..
        with open(os.path.join(FOLDER_PATH, DATA_FILES[3]), 'rb') as f:
            self.vocab_ingrs = pickle.load(f)

        self.createPairings(filepath)
        


    def createPairings(self,filepath):
        data = pd.read_csv(filepath)
        self.pairedIngr ={}
        self.pairing_scores = {}
        count_error=0
        for index, row in data.iterrows():
            if row["prediction"] > self.min_score:
                try:
                    ingr1 = ingr_norm.normalize_ingredient(row["ingr1"]).name
                    ingr2 = ingr_norm.normalize_ingredient(row["ingr2"]).name
                    self.pairing_scores[frozenset(
                        (self.vocab_ingrs.word2idx[ingr1], self.vocab_ingrs.word2idx[ingr2]))] = row["prediction"]
                    self.pairedIngr[ingr1]=self.vocab_ingrs.word2idx[ingr1]
                    self.pairedIngr[ingr2]=self.vocab_ingrs.word2idx[ingr2]
                except KeyError:
                    count_error += 1


        print(f"{len(self)} pairs in total")
        print(f"{count_error} pair(s) not added because of an absent ingredient")

    def bestPairingsFromIngr(self, ingr_id):
        """
        ex: ingr_id = 2
        returns [(frozenset({16, 2}), 0.01891073), (frozenset({129, 2}), 0.0022993684)]
        """
        ingr_pairs = {k: v for k, v in self.pairing_scores.items()
                      if ingr_id in k}
        return [(pair, self.pairing_scores[pair]) for pair in heapq.nlargest(self.top_k, ingr_pairs, key=self.pairing_scores.get)]

    def __len__(self):
        return len(self.pairing_scores)


if __name__ == "__main__":
    pairing = PairingData(filepath)
    print(pairing.pairedIngr)
    print(pairing.bestPairingsFromIngr(74))
