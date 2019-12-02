import torch
import os
import pickle
import time
import math
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


MAX_LENGTH=300


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)



def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


class Data:
    # TODO: change to dataloader
    def __init__(self,FOLDER_PATH,DATA_FILES,max_length=MAX_LENGTH):
        self.SOS_token = 0
        self.EOS_token = 1
        self.UNK_token = 2
        self.max_length = max_length

        with open(os.path.join(FOLDER_PATH,DATA_FILES[3]),'rb') as f:
            self.vocab_ingrs=pickle.load(f)
            
        with open(os.path.join(FOLDER_PATH,DATA_FILES[4]),'rb') as f:
            self.vocab_tokens=pickle.load(f)

        with open(os.path.join(FOLDER_PATH,DATA_FILES[2]),'rb') as f:
            self.data=pickle.load(f)
        
        self.process_data()
    
    def preprocess_data(self):

        pairs=[]
        for recipe in self.data:
            pairs.append([recipe["ingredients"],recipe["tokenized"]])

        length=0
        for i,pair in enumerate(pairs):
            length_t = len([item for sublist in pair[1] for item in sublist])
            
            if length_t>length:
                length=length_t

        pairs = self.filterPairs(pairs)  
        return pairs

    def process_data(self):
        self.pairs = pairs = self.preprocess_data()
        self.data = []
        for pair in pairs:
            self.data.append(self.tensorsFromPair(pair))

    def filterSinglePair(self,p):
        length=0
        for ingr in p[0]:
            if ingr not in self.vocab_ingrs.word2idx:
                return False
            
        for sent in p[1]:
            
            for word in sent:
                # TODO check how steps tokenized ? Put into vocab ???
                if word not in self.vocab_tokens.word2idx:
                    return False
            length+=len(sent)
        
        return length < MAX_LENGTH


    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterSinglePair(pair)]


    def list2idx(self,vocab, sentence):
        return [vocab.word2idx.get(word,self.UNK_token) for word in sentence]


    def tensorFromSentence(self,vocab, sentence,instructions=False):
        # TODO: padding ?
        if instructions:
            indexes=[]
            for sent in sentence:
                indexes.extend(self.list2idx(vocab, sent))
        else:
            indexes = self.list2idx(vocab, sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long)#.view(-1, 1)


    def tensorsFromPair(self,pair):
        input_tensor = self.tensorFromSentence(self.vocab_ingrs, pair[0])
        target_tensor = self.tensorFromSentence(self.vocab_tokens, pair[1],instructions=True)
        return (input_tensor, target_tensor)




def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))