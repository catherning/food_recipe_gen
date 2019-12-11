import os
import pickle


import sys
sys.path.insert(0, "D:\\Documents\\THU\\food_recipe_gen")
from recipe_1m_analysis.utils import Vocabulary, RecipesDataset,FOLDER_PATH, DATA_FILES 
from recipe_gen.seq2seq_model import Seq2seq,Seq2seqAtt,Seq2seqIngrAtt
from recipe_gen.seq2seq_utils import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = RecipesDataset(FOLDER_PATH,DATA_FILES)
    hidden_size = 128
    BATCH_SIZE = 4
    pairing_path = "D:\\Documents\\THU\\Other recipe models\\KitcheNette-master\\KitcheNette-master\\results\\prediction_unknowns_smaller_kitchenette_trained.mdl.csv"

    load = False
    # model = Seq2seqAtt(len(data.vocab_ingrs),hidden_size,len(data.vocab_tokens),BATCH_SIZE,data,device=device,savepath=os.path.join(os.getcwd(),"recipe_gen","results"))
    model = Seq2seqIngrAtt(len(data.vocab_ingrs),hidden_size,len(data.vocab_tokens),BATCH_SIZE,data,pairing_path,device=device,savepath=os.path.join(os.getcwd(),"recipe_gen","results"))

    if load:
        model.load_state_dict(torch.load(os.path.join(os.getcwd(),"recipe_gen","results","model_12-11-15-59_500")))
        model.to(device)
    else:
        model.to(device)
        model.train_process(500, print_every=50)

    model.evaluateRandomly(n=2)

    loss,output_words, attentions = model.evaluate("tomato salad beef lemon".split())
    try:
        plt.matshow(attentions[:,0,:].numpy())
    except AttributeError:
        print("No attention to show.")
        
if __name__=="__main__":
    main()