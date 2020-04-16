import math
import os
import pickle
import sys
import argparse
import time
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from recipe_1m_analysis.utils import Vocabulary


MAX_LENGTH = 100
MAX_STEP = 10
MAX_INGR = 10
FOLDER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),os.pardir,"recipe_1m_analysis", "data") 
DATA_FILES = ["allingrs_count.pkl",
              "allwords_count.pkl",
              "recipe1m_test.pkl",
              "recipe1m_vocab_ingrs.pkl",
              "recipe1m_vocab_toks.pkl",
              "recipe1m_train.pkl"]

argparser = argparse.ArgumentParser()
argparser.add_argument('--data-folder', type=str, default=FOLDER_PATH,
                    help='Dataset path')
argparser.add_argument('--vocab-ingr-file', type=str, default=DATA_FILES[3],
                    help='Dataset path')
argparser.add_argument('--vocab-tok-file', type=str, default=DATA_FILES[4],
                    help='Dataset path')
argparser.add_argument('--train-file', type=str, default=DATA_FILES[5],
                    help='Dataset path')
argparser.add_argument('--test-file', type=str, default=DATA_FILES[2],
                    help='Dataset path')
argparser.add_argument('--max-ingr', type=int, default=MAX_INGR)
argparser.add_argument('--max-length', type=int, default=MAX_LENGTH)
argparser.add_argument('--max-step', type=int, default=MAX_STEP)

class RecipesDataset(Dataset):
    """Recipes dataset."""

    def __init__(self,args,train=True):
        self.max_length = args.max_length
        self.max_ingr = args.max_ingr
        self.max_step = args.max_step
        self.model_name =  args.model_name
        self.samples_max = args.samples_max
        

        with open(os.path.join(args.data_folder,args.vocab_ingr_file),'rb') as f:
            self.vocab_ingrs=pickle.load(f)
            
        with open(os.path.join(args.data_folder,args.vocab_tok_file),'rb') as f:
            self.vocab_tokens=pickle.load(f)
        
        if self.model_name == "Seq2seqCuisinePairing":
            with open(os.path.join(args.data_folder,args.vocab_cuisine_file),'rb') as f:
                self.vocab_cuisine = pickle.load(f)

        if "Hierarchical" in self.model_name:
            self.hierarchical = True
        else:
            self.hierarchical = False

        self.PAD_token = self.vocab_ingrs.word2idx.get("<pad>",0)
        self.SOS_token = self.vocab_ingrs.word2idx.get("<sos>",1)
        self.EOS_token = self.vocab_ingrs.word2idx.get("<eos>",2)
        self.UNK_token = self.vocab_ingrs.word2idx.get("<unk>",3)

        if train:
            with open(os.path.join(args.data_folder,args.train_file),'rb') as f:
                self.data=pickle.load(f)
        else:
            with open(os.path.join(args.data_folder,args.test_file),'rb') as f:
                self.data=pickle.load(f)
                
        self.process_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

    def process_data(self):
        data = []
        if self.model_name == "Seq2seqCuisinePairing":
            count_e = 0
            for idx,recipe in self.data.items():
                try:
                    pair = [recipe["ingredients"],recipe["tokenized"],recipe["title"],idx]
                    if self.filterSinglePair(pair):
                        _dict = self.tensorsFromPair(pair)
                        _dict["cuisine"]=torch.LongTensor([self.vocab_cuisine.word2idx[recipe["cuisine"]]])
                        del _dict["title"]
                        data.append(_dict)
                except AttributeError:
                    count_e+=1
                    continue
            print("{} recipes without cuisine. {} recipes kept.".format(count_e,len(data)))

        else:
            for idx,recipe in self.data.items():
                pair = [recipe["ingredients"],recipe["tokenized"],recipe["title"],idx]
                if self.filterSinglePair(pair):
                    data.append(self.tensorsFromPair(pair))

        self.data = data[:self.samples_max]

    def filterSinglePair(self,p):
        try: 
            for ingr in p[0]:
                if ingr.name not in self.vocab_ingrs.word2idx:
                    return False
        except AttributeError:
            for ingr in p[0]:
                if ingr not in self.vocab_ingrs.word2idx:
                    return False

        if not len(p[0])>=self.max_ingr-1:
            return False

        lengths=[]
        for sent in p[1]:
            for word in sent:
                # TODO check how steps tokenized ? Put into vocab ???
                if word not in self.vocab_tokens.word2idx:
                    return False

            lengths.append(len(sent))
        
        if self.hierarchical:
            return all(l<self.max_length-1 for l in lengths)
        else:
            return sum(lengths) < self.max_length-1  # -1 because need to add eos to input and target

    def instr2idx(self, sentence): 
        # if doesn't find, use unk_token kind of useless because filtered before ?
        return torch.Tensor([self.vocab_tokens.word2idx.get(word,self.UNK_token) for word in sentence])

    def ingr2idx(self,ingr_list):
        try:
            return torch.Tensor([self.vocab_ingrs.word2idx.get(word.name,self.UNK_token) for word in ingr_list])
        except AttributeError:
            return torch.Tensor([self.vocab_ingrs.word2idx.get(word,self.UNK_token) for word in ingr_list])

    def tensorFromSentence(self,vocab, sentence,instructions=False): 
        max_size = instructions * self.max_length + (1-instructions) * self.max_ingr
        tensor_ = torch.ones(max_size,dtype=torch.long) * self.PAD_token
        if instructions:
            if self.hierarchical:
                tensor_ = torch.ones(self.max_step,max_size,dtype=torch.long) * self.PAD_token
                b_id = []
                for i,sent in enumerate(sentence):
                    if i>=MAX_STEP:
                        break
                    tokenized = self.instr2idx(sent)
                    sent_len = len(tokenized)
                    tensor_[i][:sent_len]= tokenized
                    tensor_[i][sent_len]=self.EOS_token
                    b_id.append(sent_len)            
            else:
                b_id = 0
                for sent in sentence:
                    tokenized = self.instr2idx(sent)
                    sent_len = len(tokenized)
                    tensor_[b_id:b_id+sent_len]= tokenized
                    b_id += sent_len
                tensor_[b_id]=self.EOS_token
                b_id+=1

        else:
            tokenized = self.ingr2idx(sentence)[:max_size-1]
            tensor_[:len(tokenized)]= tokenized
            tensor_[len(tokenized)]=self.EOS_token
            b_id = len(tokenized)+1

        return tensor_, b_id

    def tensorsFromPair(self,pair):
        input_tensor,_ = self.tensorFromSentence(self.vocab_ingrs, pair[0])
        target_tensor,target_length = self.tensorFromSentence(self.vocab_tokens, pair[1],instructions=True)
        title,_ = self.tensorFromSentence(self.vocab_tokens,pair[2])
        
        return {"ingr":input_tensor,
                    "target_instr": target_tensor,
                    "target_length":target_length,
                    "title":title,
                    "id":pair[-1]}
                # "ingr_tok":pair[0],
                # "target_tok":pair[1]}

def inverse_sigmoid_decay(decay_factor, i):
    return decay_factor / (
        decay_factor + math.exp(i / decay_factor))


def flattenSequence(data, lengths):
    # input has already <sos> removed for target, because decoded outputs for prediction doesn't have it.
    # this function will cut the <eos>
    arr = []
    if type(lengths) is torch.Tensor:
        for i,length in enumerate(lengths):
            arr.append(data[i,:length])
    
    else:
        for i,l_batch in enumerate(lengths):
            for j,l in enumerate(l_batch):
                arr.append(data[j,i,:l])

    return torch.cat(arr, dim=0)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def showSentAttention(input_sentence, output_words, attentions,path,name=None):
    b_id =0
    for i,sent in enumerate(" ".join(output_words).replace('<eos> ', ', ').split(" , ")):
        showAttention(input_sentence,sent,attentions[b_id:b_id+len(sent)],path,name="{}_{}".format(name,i))
        b_id = len(sent)

def showPairingAttention(comp_ingr, output_words, attentions,path,name=None):
    for i,w in enumerate(output_words):
        showAttention(comp_ingr[i],w,attentions[i].unsqueeze(0),path,name="{}_{}".format(name,i))

def showAttention(input_sentence, output_words, attentions,path,name=None):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(input_sentence.split(' '), rotation=90)
    ax.set_yticklabels([''] + output_words.split(' '))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    plt.savefig(os.path.join(path,'attention_{}.png'.format(name)))
    plt.close(fig)

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



if __name__ == "__main__":
    args = argparser.parse_args()

    with open(os.path.join(FOLDER_PATH, DATA_FILES[3]), 'rb') as f:
        vocab_ingrs = pickle.load(f)

    recipe_dataset = RecipesDataset(args)
    print(recipe_dataset[0])
    dataset_loader = torch.utils.data.DataLoader(recipe_dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)


    for i_batch, sample_batched in enumerate(dataset_loader):
        print(i_batch, sample_batched)

        if i_batch == 1:
            break