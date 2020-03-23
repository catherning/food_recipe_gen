#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from os import listdir
from os.path import isfile, join
sys.path.insert(0, os.getcwd())
import json
import pandas as pd 

from recipe_1m_analysis.ingr_normalization import normalize_ingredient

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances_argmin_min


# In[2]:


FOLDER_PATH = "F:\\user\\Google Drive\\Catherning Folder\\THU\\Thesis\\Work\\Recipe datasets\\scirep-cuisines-detail"
FILES = ["allr_recipes.txt","epic_recipes.txt","menu_recipes.txt"]
map_file = "map.txt"


# In[3]:


recipes = []
for file in FILES:
    with open(os.path.join(FOLDER_PATH,file)) as f:
        for i,recipe in enumerate(f):
            data = recipe.strip('\n').split('\t')
            list_ingr = []
            for ingr_raw in data[1:]:
                try:
                    list_ingr.append(normalize_ingredient(ingr_raw).name)
                except AttributeError:
                    continue
            recipes.append({"cuisine":data[0],"id":i,"ingredients":list(set(list_ingr))})
        
df_train = pd.DataFrame(recipes)
df_train['all_ingredients'] = df_train['ingredients'].map(";".join)


# In[4]:


df_train


# In[5]:


df_train['cuisine'].value_counts()
40150/57691


# In[6]:


df_train['cuisine'].value_counts()
df_train['cuisine'].value_counts().head(10).plot(kind='bar')


# ### Data cleaning

# In[7]:


replace_dict = {}
with open(os.path.join(FOLDER_PATH,map_file)) as f:
    for line in f:
        l = line.split()
        try:
            replace_dict[l[0]]=l[1]
        except IndexError:
            pass
replace_dict["asian"]="Asian"


# In[8]:


df_train=df_train.replace(replace_dict)
df_train['cuisine'].value_counts()

# Fusion of the same cuisines
#df_train=df_train.replace({"Mexico":"Mexican",
                        #    "mexico":"Mexican",
                        #    "chinese":"Chinese",
                        #    "China":"Chinese",
                        #    "France":"French",
                        #    "japanese":"Japanese",
                        #    "Japan":"Japanese",
                        #    "Thailand":"Thai",
                        #    "German":"Germany",
                        #    "India":"Indian",
                        #    "Israel":"Jewish",
                        #    "italian":"Italian",
                        #    "Italy":"Italian",
                        #    "Scandinavia":"Scandinavian",
                        #    "Vietnam":"Vietnamese",
                        #    "Korea":"Korean",
                        #    "korean":"Korean",
                        #    "EasternEuropean_Russian":"Eastern-Europe",
                        #    'Spain':'Spanish_Portuguese'})
df_train
# ## Removing cuisines with not enough recipes

# In[9]:


RECIPE_THRESHOLD=10
cuisine_count= df_train['cuisine'].value_counts()
to_drop = [cuisine_count[cuisine_count == el].index[0] for el in cuisine_count if el<RECIPE_THRESHOLD]
df_train = df_train[~df_train["cuisine"].isin(to_drop)]


# In[10]:


len(df_train)


# In[11]:


# Saving the cleaned data
df_train.to_pickle(os.path.join(FOLDER_PATH,"full_data.pkl"))


# In[12]:


df_train


# In[ ]:




