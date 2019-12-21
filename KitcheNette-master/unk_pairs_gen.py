import os
import sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import pickle
import csv

from recipe_gen.seq2seq_utils import FOLDER_PATH, DATA_FILES
from recipe_1m_analysis.utils import Vocabulary
import recipe_1m_analysis.ingr_normalization as ingr_norm


known_file = os.path.join(os.getcwd(),"KitcheNette-master","data","kitchenette_pairing_scores.csv")
with open(os.path.join(known_file),'rb') as f:
    known_pairs=pd.read_csv(f)

known_dict = {}
count_e=0
for row_idx, row in known_pairs.iterrows():
    try:
        ingr1_n = ingr_norm.normalize_ingredient(row["ingr1"])
        ingr2_n = ingr_norm.normalize_ingredient(row["ingr2"])
        known_dict[frozenset((ingr1_n.name,ingr2_n.name))]=row["npmi"]
    except AttributeError:
        count_e+=1

print(f"Removed {count_e} false pairings")

with open(os.path.join(FOLDER_PATH,DATA_FILES[3]),'rb') as f:
    vocab_ingrs=pickle.load(f)

vocab_main_ingr = Vocabulary()
for k,v in vocab_ingrs.idx2word.items():
    main_ingr=None
    min_len=100
    for ingr in v:
        if len(ingr.split("_"))<min_len:
            main_ingr = ingr
            min_len = len(ingr.split("_"))
    try:
        vocab_main_ingr.add_word(ingr_norm.normalize_ingredient(main_ingr).name,k)
    except:
        print(main_ingr)

#TODO: remove special tokens sos, eos, pad, unk more properly
del vocab_main_ingr.word2idx["<"]
del vocab_main_ingr.idx2word[0]

list_ingr = list(vocab_main_ingr.word2idx.keys()) 

with open('main_pairing_prediction.csv', mode='w',newline='') as f:
    csvwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(["ingr1","ingr2"])
    for i,ingr1 in enumerate(list_ingr):
        for ingr2 in list_ingr[i+1:]:
            if frozenset((ingr1,ingr2)) not in known_dict:
                csvwriter.writerow([ingr1,ingr2])


