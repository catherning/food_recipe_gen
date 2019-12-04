import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

###
# Add following lines in the file where you want to import the current utils file 
# import sys
# sys.path.insert(0, "D:\\Documents\\food_recipe_gen\\recipe_1m_analysis")
###

MAX_LENGTH = 300
MAX_INGR = 10
FOLDER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
DATA_FILES = ["allingrs_count.pkl",
              "allwords_count.pkl",
              "recipe1m_test.pkl",
              "recipe1m_vocab_ingrs.pkl",
              "recipe1m_vocab_toks.pkl"]


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<pad>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def add_word(self, word, idx=None):
        if idx is None:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
            return self.idx
        else:
            if word not in self.word2idx:
                self.word2idx[word] = idx
                if idx in self.idx2word.keys():
                    self.idx2word[idx].append(word)
                else:
                    self.idx2word[idx] = [word]

                return idx


class RecipesDataset(Dataset):
    """Recipes dataset."""

    def __init__(self,FOLDER_PATH,DATA_FILES,max_ingr=10,max_length=MAX_LENGTH):
        self.SOS_token = 0
        self.EOS_token = 1
        self.UNK_token = 2 # TODO: use the predefined special tokens ids from the vocabs!!!!
        self.max_length = max_length
        self.max_ingr = max_ingr

        with open(os.path.join(FOLDER_PATH,DATA_FILES[3]),'rb') as f:
            self.vocab_ingrs=pickle.load(f)
            
        with open(os.path.join(FOLDER_PATH,DATA_FILES[4]),'rb') as f:
            self.vocab_tokens=pickle.load(f)

        with open(os.path.join(FOLDER_PATH,DATA_FILES[2]),'rb') as f:
            self.data=pickle.load(f)
        
        self.process_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]
    
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
        
        return length < self.max_length and len(p[0])<self.max_ingr


    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterSinglePair(pair)]


    def list2idx(self,vocab, sentence): 
        # if doesn't find, use unk_token kind of useless because filtered before
        return torch.Tensor([vocab.word2idx.get(word,self.UNK_token) for word in sentence])


    def tensorFromSentence(self,vocab, sentence,instructions=False):
        max_size = instructions * self.max_length + (1-instructions) * self.max_ingr
        tensor_ = torch.zeros(max_size,dtype=torch.long) 
        if instructions:
            indexes=[]
            b_id=0
            for sent in sentence:
                tokenized = self.list2idx(vocab, sent)
                tensor_[b_id:b_id+len(tokenized)]= tokenized
                b_id += len(tokenized)
        else:
            tokenized = self.list2idx(vocab, sentence)
            tensor_[:len(tokenized)]= tokenized
            b_id = len(tokenized)

        tensor_[b_id]=self.EOS_token
        return tensor_


    def tensorsFromPair(self,pair):
        input_tensor = self.tensorFromSentence(self.vocab_ingrs, pair[0])
        target_tensor = self.tensorFromSentence(self.vocab_tokens, pair[1],instructions=True)
        return (input_tensor, target_tensor)


if __name__ == "__main__":
    recipe_dataset = RecipesDataset(FOLDER_PATH, DATA_FILES,max_ingr=MAX_INGR,max_length=MAX_LENGTH)
    dataset_loader = torch.utils.data.DataLoader(recipe_dataset,
                                                 batch_size=32, shuffle=True,
                                                 num_workers=4)

    with open(os.path.join(FOLDER_PATH, DATA_FILES[3]), 'rb') as f:
        vocab_ingrs = pickle.load(f)

    recipe = recipe_dataset[0]

    for i_batch, sample_batched in enumerate(dataset_loader):
        print(i_batch, sample_batched[0],
          sample_batched[1])

        if i_batch == 1:
            break

    # one_hot_enc = torch.nn.functional.one_hot(torch.LongTensor(ingr_idx), max(vocab_ingrs.idx2word.keys()))
    # print(one_hot_enc.shape)
