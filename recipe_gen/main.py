import os
import pickle
import sys
import argparse
import logging
from datetime import datetime
import pathlib
import importlib
sys.path.insert(0, os.getcwd())
from recipe_1m_analysis.utils import Vocabulary
from recipe_gen.seq2seq_utils import *
from recipe_gen.pairing_utils import PairingData

LOGGER = logging.getLogger()
PAIRING_PATH = os.path.join(os.getcwd(),"KitcheNette-master","results","pairings.pkl")
SAVING_PATH = os.path.join(os.getcwd(), "recipe_gen", "results")
# DATA_FILES = ["allingrs_count.pkl",
#               "allwords_count.pkl",
#               "recipe1m_test.pkl",
#               "recipe1m_vocab_ingrs.pkl",
#               "recipe1m_vocab_toks.pkl",
#               "recipe1m_train.pkl"]

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
argparser.add_argument('--data-folder', type=str, default=FOLDER_PATH,
                       help='Dataset path')
argparser.add_argument('--pairing-path', type=str, default=PAIRING_PATH,
                       help='Dataset path')
argparser.add_argument('--saving-path', type=str, default=SAVING_PATH,
                       help='Dataset path')
argparser.add_argument('--vocab-ingr-file', type=str, default=DATA_FILES[3],
                       help='Dataset path')
argparser.add_argument('--vocab-tok-file', type=str, default=DATA_FILES[4],
                       help='Dataset path')
argparser.add_argument('--train-file', type=str, default=DATA_FILES[5],
                       help='Dataset path')
argparser.add_argument('--test-file', type=str, default=DATA_FILES[2],
                       help='Dataset path')

# Run settings
argparser.add_argument('--model-name', type=str, choices=['Seq2seq','Seq2seqAtt',
                       'Seq2seqIngrAtt','Seq2seqIngrPairingAtt','Seq2seqTitlePairing'],
                       default="Seq2seqIngrPairingAtt",
                       help='Model name for saving/loading')
argparser.add_argument('--title', type='bool', nargs='?',  #XXX: for now, need to put title arg AND model-name Seq2seqTitlePairing
                        const=True, default=True,
                       help='Title input')
argparser.add_argument('--print-step', type=int, default=50,
                       help='Display steps')
argparser.add_argument('--validation-step', type=int, default=1,
                       help='Number of random search validation')
argparser.add_argument('--train-mode', type='bool', nargs='?',
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
argparser.add_argument('--epoch', type=int, default=100)
argparser.add_argument('--n-iters', type=int, default=500)
argparser.add_argument('--learning-rate', type=float, default=0.01)
argparser.add_argument('--teacher-forcing-ratio', type=float, default=1)
argparser.add_argument('--dropout', type=float, default=0.1)
argparser.add_argument('--topk', type=float, default=3)

# Model config
argparser.add_argument('--hidden-size', type=int, default=128)
argparser.add_argument('--max-ingr', type=int, default=10)
argparser.add_argument('--max-length', type=int, default=100)

argparser.add_argument('--seed', type=int, default=3)

args = argparser.parse_args()

def init_logging(args):
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

    # Create save folder
    args.saving_path = saving_path = os.path.join(args.saving_path, args.model_name,datetime.now().strftime('%m-%d-%H-%M'))
    if not os.path.isdir(saving_path):
        pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)
        print('...created '+ saving_path)

    logfile = logging.FileHandler(os.path.join(saving_path,'log.txt'), 'w')

    logfile.setFormatter(fmt)
    LOGGER.addHandler(logfile)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed

def getDefaultArgs(argparser):
    all_defaults = {}
    for key in vars(args):
        all_defaults[key] = argparser.get_default(key)
    return all_defaults

def main():
    init_logging(args)
    args.seed=init_seed(args.seed)
    args.logger=LOGGER
    args.defaults=getDefaultArgs(argparser)

    model_class=getattr(importlib.import_module("recipe_gen.seq2seq_model"), args.model_name)
    model = model_class(args)

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(
            os.getcwd(), "recipe_gen", "results",args.model_name, "best_model")))#XXX:lack date of run
        model.to(args.device)
        print("Model loaded.")
    else:
        model.to(args.device)
        print("Begin training.")
        model.train_process()

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
