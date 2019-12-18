import os
import pickle
import sys
sys.path.insert(0, os.getcwd())

from recipe_1m_analysis.utils import Vocabulary
from recipe_gen.seq2seq_model import *
from recipe_gen.seq2seq_utils import *



def main():
    max_length = 100
    hidden_size = 128
    BATCH_SIZE = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = RecipesDataset(FOLDER_PATH, DATA_FILES, max_length=max_length)
    # pairing_path = "D:\\Documents\\THU\\Other recipe models\\KitcheNette-master\\KitcheNette-master\\results\\prediction_unknowns_smaller_kitchenette_trained.mdl.csv"
    pairing_path = "G:\\Other files\\KitcheNette-master\\results\\prediction_unknowns_smaller_kitchenette_trained.mdl.csv"
    load = False
    # model = Seq2seq(len(data.vocab_ingrs), hidden_size, len(data.vocab_tokens), BATCH_SIZE, data, device=device,
    #                   savepath=os.path.join(os.getcwd(), "recipe_gen", "results"), teacher_forcing_ratio=1, max_length=max_length)

    model = Seq2seqAtt(len(data.vocab_ingrs), hidden_size, len(data.vocab_tokens), BATCH_SIZE, data, device=device,
                      savepath=os.path.join(os.getcwd(), "recipe_gen", "results"), teacher_forcing_ratio=1, max_length=max_length)
    model = Seq2seqIngrAtt(len(data.vocab_ingrs), hidden_size, len(data.vocab_tokens), BATCH_SIZE, data, 
                        device=device, savepath=os.path.join(os.getcwd(), "recipe_gen", "results"), teacher_forcing_ratio=1, max_length=max_length)
    model = Seq2seqIngrPairingAtt(len(data.vocab_ingrs), hidden_size, len(data.vocab_tokens), BATCH_SIZE, data, pairing_path, 
                        device=device, savepath=os.path.join(os.getcwd(), "recipe_gen", "results"), teacher_forcing_ratio=1, max_length=max_length)

    if load:
        model.load_state_dict(torch.load(os.path.join(
            os.getcwd(), "recipe_gen", "results", "model_12-17-00-26_100")))
        model.to(device)
        print("Model loaded.")
    else:
        model.to(device)
        print("Begin training.")
        model.train_process(500, print_every=50)

    model.evaluateRandomly(n=2)

    _, output_words, attentions = model.evaluate(
        "tomato salad beef lemon".split())
    print(' '.join(output_words[0]))
    try:
        plt.matshow(attentions[:, 0, :].numpy())
    except (TypeError, AttributeError):
        print("No attention to show.")


if __name__ == "__main__":
    main()
