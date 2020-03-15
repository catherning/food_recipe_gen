import csv
import heapq
import os
import pickle
import sys

import pandas as pd
sys.path.append(os.getcwd())

import recipe_1m_analysis.ingr_normalization as ingr_norm
from recipe_1m_analysis.utils import Vocabulary
import recipe_1m_analysis.utils as utils
from recipe_gen.seq2seq_utils import DATA_FILES, FOLDER_PATH

known_path = os.path.join(os.getcwd(),"KitcheNette-master","data","kitchenette_pairing_scores.csv")
filepath = os.path.join(os.getcwd(),"KitcheNette-master","results","prediction_unknowns_kitchenette_pretrained.mdl.csv")
pickle_path = "./KitcheNette-master/results/pairings.pkl"

class PairingData:
    def __init__(self, filepaths, pickle_file=pickle_path, min_score=0, top_k=20,trim=True):
        self.min_score = min_score
        self.top_k = top_k
        self.pairedIngr ={}
        self.pairing_scores = {}
        # XXX:repeat with other data classes..
        with open(os.path.join(FOLDER_PATH, DATA_FILES[3]), 'rb') as f:
            self.vocab_ingrs = pickle.load(f)

        self.createPairings(filepaths[0])
        if len(filepaths)==2:
            self.createPairings(filepaths[1],unknown=False)
        
        if trim:
            self.trimDict()

        self.toPickle(pickle_file)

    def addPairing(self,ingr1,ingr2,score):
        try:
            self.pairing_scores[self.pairedIngr[ingr1]][self.pairedIngr[ingr2]] = score
        except KeyError:
            self.pairing_scores[self.pairedIngr[ingr1]] = {self.pairedIngr[ingr2]:score}

    def trimDict(self):
        for v in self.pairing_scores.values():
            to_delete = heapq.nlargest(len(v)-self.top_k, v, key=v.get)
            for idx in to_delete:
                del v[idx]

    def createPairings(self,filepath,unknown=True):
        if unknown:
            score_name = "prediction"
        else:
            score_name = "npmi"

        data = pd.read_csv(filepath)
        count_error=0
        for index, row in data.iterrows():
            if row[score_name] > self.min_score:
                try:
                    ingr1 = ingr_norm.normalize_ingredient(row["ingr1"]).name
                    ingr2 = ingr_norm.normalize_ingredient(row["ingr2"]).name
                    if ingr1 not in self.pairedIngr:
                        self.pairedIngr[ingr1]=self.vocab_ingrs.word2idx[ingr1]
                    if ingr2 not in self.pairedIngr:
                        self.pairedIngr[ingr2]=self.vocab_ingrs.word2idx[ingr2]
                except (KeyError,AttributeError):
                    print(ingr1,ingr2)
                    count_error += 1
                    continue
                
                self.addPairing(ingr1,ingr2,row[score_name])
                self.addPairing(ingr2,ingr1,row[score_name])


        print("{} pairs in total".format(len(self)))
        print("{} pair(s) not added because of an absent ingredient in the vocab or false ingredient".format(count_error))

    def bestPairingsFromIngr(self, ingr_id):
        """
        ex: ingr_id = 2
        returns [(frozenset({16, 2}), 0.01891073), (frozenset({129, 2}), 0.0022993684)]
        """
        try:
            return  [(ingr,score) for ingr,score in self.pairing_scores[ingr_id.item()].items()]
        except KeyError:
            return []

    def __len__(self):
        return len(self.pairing_scores)
    
    def toPickle(self,picklefile):
        with open(picklefile, 'wb') as f:
            pickle.dump(self, f)
    

if __name__ == "__main__":

    pairing = PairingData([filepath,known_path])
    print(len(pairing.pairedIngr))
    # print(pairing.bestPairingsFromIngr(74))
    
    with open(pickle_path,'rb') as f:
        data=pickle.load(f)
    print(len(data.pairedIngr))