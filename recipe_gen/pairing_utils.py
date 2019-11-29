import os
import csv
import pickle
import heapq
import pandas as pd
from recipe_1m_analysis.utils import Vocabulary, FOLDER_PATH, DATA_FILES

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
        for index, row in data.iterrows():
            if row["prediction"] > min_score:
                try:
                    self.pairing_scores[frozenset(
                        (vocab_ingrs.word2idx[row["ingr1"]], vocab_ingrs.word2idx[row["ingr2"]]))] = row["prediction"]
                except KeyError:
                    pass

    def bestPairingsFromIngr(self, ingr_id):
        ingr_pairs = {k: v for k, v in self.pairing_scores.items()
                      if ingr_id in k}
        return [(pair, self.pairing_scores[pair]) for pair in heapq.nlargest(self.top_k, ingr_pairs, key=self.pairing_scores.get)]


if __name__ == "__main__":
    pairing = PairingData(filepath)

    pairing.bestPairingsFromIngr(2)
