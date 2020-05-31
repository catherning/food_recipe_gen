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
        data_nn=pickle.load(f)
        
      with open(os.path.join(DATA_FOLDER,"recipe1m_{}_cuisine_log_reg.pkl".format(file)),'rb') as f:
        data_ml=pickle.load(f)
        
      full_data = dict(filter(lambda rec: data_ml[rec[0]]["cuisine"]==rec[1]["cuisine"], data_nn.items()))
      with open(os.path.join(folder, 'recipe1m_{}_cuisine.pkl'.format(file)), 'wb') as f:
        pickle.dump(full_data, f)
        
      for k,sp in split.items():
        new_data = dict(filter(lambda rec: rec[1]["cuisine"] in sp, full_data.items()))
        with open(os.path.join(folder, 'recipe1m_{}_{}.pkl'.format(file,k)), 'wb') as f:
            pickle.dump(new_data, f)

createZeroShotDataset(DATA_FOLDER)
