#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
import os
import sys
sys.path.append(os.getcwd())


DATA_FOLDER = os.path.join(os.getcwd(),"recipe_1m_analysis", "data") 
FILE = ["train","test","val"] 
file = FILE[2]
KEEP_CUISINE = {"NorthAmerican","NorthernEuropean","WesternEuropean","SouthernEuropean","EasternEuropean"}


with open(os.path.join(DATA_FOLDER,"recipe1m_{}_cuisine_nn.pkl".format(file)),'rb') as f:
    data=pickle.load(f)

def createZeroShotDataset(folder):
    file = FILE[0]
    new_data = dict(filter(lambda rec: rec[1]["cuisine"] in KEEP_CUISINE,data.items()))
    with open(os.path.join(folder, 'recipe1m_{}_ZeroShot_nn.pkl'.format(file)), 'wb') as f:
        pickle.dump(new_data, f)
    
    for file in FILE[1:]:
        new_data = dict(filter(lambda rec: rec[1]["cuisine"] not in KEEP_CUISINE,data.items()))
        with open(os.path.join(folder, 'recipe1m_{}_ZeroShot_nn.pkl'.format(file)), 'wb') as f:
            pickle.dump(new_data, f)

createZeroShotDataset(DATA_FOLDER)

