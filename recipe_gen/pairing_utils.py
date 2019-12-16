import os
import csv
import pickle
import heapq
import pandas as pd
import sys
sys.path.insert(0, "D:\\Documents\\THU\\food_recipe_gen")
from recipe_1m_analysis.utils import Vocabulary
import recipe_1m_analysis.ingr_normalization as ingr_norm
from recipe_gen.seq2seq_utils import FOLDER_PATH, DATA_FILES

filepath = "D:\\Documents\\THU\\Other recipe models\\KitcheNette-master\\KitcheNette-master\\results\\prediction_unknowns_smaller_kitchenette_trained.mdl.csv"


class PairingData:
    def __init__(self, filepath, min_score=0, top_k=20):

        self.min_score = min_score
        self.top_k = top_k
        # XXX:repeat with other data classes..
        with open(os.path.join(FOLDER_PATH, DATA_FILES[3]), 'rb') as f:
            self.vocab_ingrs = pickle.load(f)

        data = pd.read_csv(filepath)
        self.pairing_scores = {}
        count_error=0
        # TODO: norm ingr ingr_norm.normalize_ingredient(row["ingr1"]) before 2idx 
        for index, row in data.iterrows():
            if row["prediction"] > min_score:
                try:
                    self.pairing_scores[frozenset(
                        (self.vocab_ingrs.word2idx[ingr_norm.normalize_ingredient(row["ingr1"]).name], self.vocab_ingrs.word2idx[ingr_norm.normalize_ingredient(row["ingr2"]).name]))] = row["prediction"]
                except KeyError:
                    count_error+=1
                # except AttributeError:
                #     print(row["ingr1"],ingr_norm.normalize_ingredient(row["ingr1"]))
                #     print(row["ingr2"],ingr_norm.normalize_ingredient(row["ingr2"]))

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
    
    print(pairing.bestPairingsFromIngr(2))