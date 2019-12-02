import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

###
# Add following lines in the file where you want to import the current utils file 
# import sys
# sys.path.insert(0, "D:\\Documents\\food_recipe_gen\\recipe_1m_analysis")
###

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

    def __init__(self, file):
        """
        Args:
            file (string): Path to the file
        """
        with open(file, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'title': self.data[idx]["title"],
                  'ingredients': self.data[idx]["ingredients"],
                  'instructions': self.data[idx]["tokenized"], }

        return sample


if __name__ == "__main__":

    recipe_dataset = RecipesDataset(os.path.join(FOLDER_PATH, DATA_FILES[2]))
    dataset_loader = torch.utils.data.DataLoader(recipe_dataset,
                                                 batch_size=32, shuffle=True,
                                                 num_workers=4)

    with open(os.path.join(FOLDER_PATH, DATA_FILES[3]), 'rb') as f:
        vocab_ingrs = pickle.load(f)

    recipe = recipe_dataset[0]
    ingr = recipe["ingredients"]
    ingr_idx = []
    for el in ingr:
        ingr_idx.append(vocab_ingrs.word2idx[el])

    one_hot_enc = torch.nn.functional.one_hot(torch.LongTensor(ingr_idx), max(vocab_ingrs.idx2word.keys()))
    print(one_hot_enc.shape)
