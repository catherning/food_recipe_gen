#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from joblib import dump, load
import pickle
import sys
sys.path.insert(0, os.getcwd())

from collections import Counter
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.lancaster import LancasterStemmer
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


FILES  = ["train","val","test"]
DATA_FOLDER = os.path.join(os.getcwd(),"recipe_1m_analysis","data")
MODEL_FOLDER = os.path.join(os.getcwd(),"cuisine_classification","ml_results")

svc = load(os.path.join(MODEL_FOLDER,"Logistic Regression.joblib"))
enc = load(os.path.join(MODEL_FOLDER,"scikit_vocab_cuisine.joblib"))

for file in FILES:
    with open(os.path.join(DATA_FOLDER,"recipe1m_"+file+".pkl"),"rb") as f:
        data = pickle.load(f)

    ingr_input = [";".join([ingr.name for ingr in recipe["ingredients"]]) for recipe in data.values() ]

    vocab = load(os.path.join(MODEL_FOLDER,"scikit_vocab.joblib"))

    cv = CountVectorizer(vocabulary=vocab)
    X = cv.fit_transform(ingr_input)

    prediction = svc.predict(X)
    str_pred =enc.inverse_transform(prediction)

    counter_svc=Counter(str_pred)


    for i,v in enumerate(data.values()):
        v["cuisine"]=str_pred[i]

    with open(os.path.join(DATA_FOLDER,"recipe1m_"+file+"_cuisine_log_reg.pkl"), "wb") as f:
        pickle.dump(data, f)
