import os
import pickle


import sys
sys.path.insert(0, "D:\\Documents\\THU\\food_recipe_gen")
from recipe_1m_analysis.utils import Vocabulary, RecipesDataset,FOLDER_PATH, DATA_FILES 
from recipe_gen.seq2seq_model import Seq2seq,Seq2seqAtt
from recipe_gen.seq2seq_utils import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = RecipesDataset(FOLDER_PATH,DATA_FILES)
    hidden_size = 256
    BATCH_SIZE = 16

    model = Seq2seqAtt(len(data.vocab_ingrs.idx2word.keys()),hidden_size,len(data.vocab_tokens.idx2word.keys()),BATCH_SIZE,data,device=device)
    model.to(device)
    model.train_process(20, print_every=100)


    model.evaluateRandomly(n=2)

    loss,output_words, attentions = model.evaluate("tomato salad beef lemon".split())
    plt.matshow(attentions.numpy())

if __name__=="__main__":
    main()