import os
import pickle
import sys
import argparse
import logging
import importlib
sys.path.insert(0, os.getcwd())
from recipe_1m_analysis.utils import Vocabulary
from recipe_gen.seq2seq_utils import *
from recipe_gen.pairing_utils import PairingData

LOGGER = logging.getLogger()
PAIRING_PATH = os.path.join(os.getcwd(),"KitcheNette-master","results","pairings.pkl")
SAVING_PATH = os.path.join(os.getcwd(), "recipe_gen", "results")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

argparser = argparse.ArgumentParser()
argparser.register('type', 'bool', str2bool)

# Directories
argparser.add_argument('--data-path', type=str, default=FOLDER_PATH,
                       help='Dataset path')
argparser.add_argument('--pairing-path', type=str, default=PAIRING_PATH,
                       help='Dataset path')
argparser.add_argument('--saving-path', type=str, default=SAVING_PATH,
                       help='Dataset path')

# Run settings
argparser.add_argument('--model-name', type=str, choices=['Seq2seq','Seq2seqAtt',
                       'Seq2seqIngrAtt','Seq2seqIngrPairingAtt'],
                       default="Seq2seqIngrPairingAtt",
                       help='Model name for saving/loading')
argparser.add_argument('--print-step', type=int, default=50,
                       help='Display steps')
argparser.add_argument('--validation-step', type=int, default=1,
                       help='Number of random search validation')
argparser.add_argument('--train', type='bool', nargs='?',
                        const=True, default=True,
                       help='Enable training')
argparser.add_argument('--pretrain', type='bool', nargs='?',
                        const=True, default=False,
                       help='Enable training')
argparser.add_argument('--valid', type='bool', nargs='?',
                        const=True, default=True,
                       help='Enable validation')
argparser.add_argument('--test', type='bool', nargs='?',
                        const=True, default=True,
                       help='Enable testing')
argparser.add_argument('--resume', type='bool', nargs='?',
                        const=True, default=False,
                       help='Resume saved model')
argparser.add_argument('--device', type=int, default=0,
                       help='GPU device number')

# Save outputs
argparser.add_argument('--save-embed', type='bool', nargs='?',
                        const=True, default=False,
                       help='Save embeddings with loaded model')
argparser.add_argument('--save-prediction', type='bool', nargs='?',
                        const=True, default=False,
                       help='Save predictions with loaded model')
argparser.add_argument('--save-prediction-unknowns', type='bool', nargs='?',
                        const=True, default=False,
                       help='Save pair scores with loaded model')
argparser.add_argument('--embed-d', type=int, default=1,
                       help='0:val task data, 1:v0.n data')

# Train config
argparser.add_argument('--batch-size', type=int, default=8)
argparser.add_argument('--epoch', type=int, default=200)
argparser.add_argument('--learning-rate', type=float, default=0.01)
argparser.add_argument('--teacher-forcing-ratio', type=float, default=1)
argparser.add_argument('--dropout', type=float, default=0.1)

# Model config
argparser.add_argument('--hidden-size', type=int, default=128)
argparser.add_argument('--max-ingr', type=int, default=10)
argparser.add_argument('--max-length', type=int, default=100)

argparser.add_argument('--seed', type=int, default=3)

args = argparser.parse_args()

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    init_seed(args.seed)
    data = RecipesDataset(args, DATA_FILES)
    model_class=getattr(importlib.import_module("recipe_gen.seq2seq_model"), args.model_name)
    model = model_class(args,len(data.vocab_ingrs), len(data.vocab_tokens), data)

    if args.resume:
        # TODO: save best models, split folders by model type. Log model infos
        model.load_state_dict(torch.load(os.path.join(
            os.getcwd(), "recipe_gen", "results", "model_12-17-00-26_100")))
        model.to(args.device)
        print("Model loaded.")
    else:
        model.to(args.device)
        print("Begin training.")
        model.train_process(args.epoch, print_every=args.print_step)

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
