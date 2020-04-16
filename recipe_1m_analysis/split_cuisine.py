#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import sys
sys.path.append(os.getcwd())


DATA_FOLDER = os.path.join(os.getcwd(),"recipe_1m_analysis", "data") 
FILE = ["train","test","val"] 

with open(os.path.join(DATA_FOLDER,"vocab_cuisine.pkl"),'rb') as f:
    vocab_cuisine = pickle.load(f)

KEEP_CUISINE = {"NorthAmerican","NorthernEuropean","WesternEuropean","SouthernEuropean","EasternEuropean"}
not_keep = {cuis for cuis in vocab_cuisine.word2idx.keys() if cuis not in KEEP_CUISINE}
split = {"main": KEEP_CUISINE, "rest" : not_keep}


def createZeroShotDataset(folder):
    for file in FILE:
      with open(os.path.join(DATA_FOLDER,"recipe1m_{}_cuisine_nn.pkl".format(file)),'rb') as f:
        data=pickle.load(f)
        
      for k,sp in split.items():
        new_data = dict(filter(lambda rec: rec[1]["cuisine"] in sp,data.items()))
        with open(os.path.join(folder, 'recipe1m_{}_nn_{}.pkl'.format(file,k)), 'wb') as f:
            pickle.dump(new_data, f)

createZeroShotDataset(DATA_FOLDER)
