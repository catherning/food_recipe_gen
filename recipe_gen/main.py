import argparse
import importlib
import logging
import os
import pathlib
import pickle
import sys
import torch
import time
import numpy as np
from datetime import datetime
import matplotlib as plt
import cProfile

sys.path.insert(0, os.getcwd())

from recipe_1m_analysis.utils import Vocabulary
from recipe_gen.pairing_utils import PairingData
from recipe_gen.seq2seq_utils import FOLDER_PATH,DATA_FILES, showPlot

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

PAIRING_PATH = os.path.join(
    os.getcwd(), "KitcheNette_master", "results", "pairings.pkl")
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
                       help='Pairing path')
argparser.add_argument('--saving-path', type=str, default=SAVING_PATH,
                       help='Saving path')
argparser.add_argument('--load-folder', type=str,
                       help='Loading best model in folder load-path')
argparser.add_argument('--vocab-ingr-file', type=str, default=DATA_FILES[3],
                       help='Dataset path')
argparser.add_argument('--vocab-tok-file', type=str, default=DATA_FILES[4],
                       help='Dataset path')
argparser.add_argument('--vocab-cuisine-file', type=str, default="vocab_cuisine.pkl",
                       help='Dataset path')
argparser.add_argument('--train-file', type=str, default=DATA_FILES[5],
                       help='Dataset path')
argparser.add_argument('--test-file', type=str, default=DATA_FILES[2],
                       help='Dataset path')
argparser.add_argument('--sample-id', type=str,
                       help='Sample id of the recipe to generate')

# Run settings
argparser.add_argument('--model-name', type=str, choices=[
                       'Seq2seq', 'Seq2seqAtt', 'Seq2seqIngrAtt', 'Seq2seqIngrPairingAtt',
                       'Seq2seqTitlePairing', 'Seq2seqCuisinePairing', 'Seq2seqTrans',
                       'HierarchicalSeq2seq', 'HierarchicalAtt', 'HierarchicalIngrAtt',
                       'HierarchicalIngrPairingAtt','HierarchicalTitlePairing','HierarchicalCuisinePairing'],
                       default="Seq2seqIngrPairingAtt",
                       help='Model name for saving/loading')
argparser.add_argument('--print-step', type=int, default=50,
                       help='Display steps')
argparser.add_argument('--train-mode', type='bool', nargs='?',
                       const=True, default=True,
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
argparser.add_argument('--load', type='bool', nargs='?',
                       const=True, default=False,
                       help='Load model for testing and inference')
argparser.add_argument('--device', type=int, default=0,
                       help='GPU device number')

# Train config
argparser.add_argument('--batch-size', type=int, default=8)
argparser.add_argument('--update-step', type=int, default=8)
argparser.add_argument('--samples-max', type=int, default=50000)
argparser.add_argument('--begin-epoch', type=int, default=1)
argparser.add_argument('--epoch', type=int, default=100)
argparser.add_argument('--n-iters', type=int, default=500)
argparser.add_argument('--learning-rate', type=float, default=0.001)
argparser.add_argument('--decay-factor', type=float, default=500.,
                       help='Speed of increasing the probability of sampling from model. Default: 500.')
argparser.add_argument('--temperature', type=float, default=0.7)
argparser.add_argument('--scheduled-sampling', type='bool', nargs='?',
                       const=True, default=True,
                       help='Uses scheduled-sampling')

# Model config
argparser.add_argument('--topk', type=int, default=3)
argparser.add_argument('--topp', type=float, default=0.9)
argparser.add_argument('--nucleus-sampling', type='bool', nargs='?',
                       const=True, default=True,
                       help='Uses nucleus sampling')
argparser.add_argument('--dropout', type=float, default=0.2)
argparser.add_argument('--bidirectional', type='bool', nargs='?',
                       const=True, default=True,
                       help='Uses nucleus sampling')
argparser.add_argument('--num-gru-layers', type=int, default=1)
argparser.add_argument('--unk-temp', type=float, default=0.2)

argparser.add_argument('--hidden-size', type=int, default=128)
argparser.add_argument('--word-embed', type=int, default=200)
argparser.add_argument('--ingr-embed', type=int, default=100)
argparser.add_argument('--title-embed', type=int, default=100)
argparser.add_argument('--cuisine-embed', type=int, default=30)
argparser.add_argument('--max-ingr', type=int, default=10)
argparser.add_argument('--max-length', type=int, default=100)
# for hierarchical, then max-length <30 is better, per step
argparser.add_argument('--max-step', type=int, default=10)

argparser.add_argument('--seed', type=int, default=3)


def init_logging(args):
    args.logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    args.logger.addHandler(console)

    # Create save folder
    args.saving_path = saving_path = os.path.join(
        args.saving_path, args.model_name, datetime.now().strftime('%m-%d-%H-%M'))
    if not os.path.isdir(saving_path):
        pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)

    logfile = logging.FileHandler(os.path.join(saving_path, 'log.txt'), 'w')

    logfile.setFormatter(fmt)
    args.logger.addHandler(logfile)
    args.logger.handlers = args.logger.handlers[:2]


def init_seed(args):
    if args.seed is None:
        args.seed = int(round(time.time() * 1000)) % 10000

    args.logger.info("seed = {}, pid = {}".format(args.seed, os.getpid()))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def getDefaultArgs(argparser):
    args = argparser.parse_args()
    all_defaults = {}
    for key in vars(args):
        all_defaults[key] = argparser.get_default(key)
    args.defaults = all_defaults
    return args


def main():
    args = getDefaultArgs(argparser)
    args.logger = LOGGER = logging.getLogger()
    init_logging(args)
    init_seed(args)

    try:
        model_class = getattr(importlib.import_module(
            "recipe_gen.seq2seq_model"), args.model_name)
    except AttributeError:
        model_class = getattr(importlib.import_module(
            "recipe_gen.hierarchical_model"), args.model_name)

    model = model_class(args)

    if args.resume:
        path = os.path.join(
            os.getcwd(), "recipe_gen", "results", args.model_name,args.load_folder)
        newest_file = max(filter(lambda f: f[0]=="t",os.listdir(path)), key=lambda f: os.path.getmtime(os.path.join(path,f)))
        
        try:
            checkpoint = torch.load(os.path.join(path, newest_file))
        except RuntimeError:
            checkpoint = torch.load(os.path.join(path, newest_file),map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        args.begin_epoch = checkpoint['epoch']+1
        model.training_losses = checkpoint['train_losses']
        model.val_losses = checkpoint['val_losses']
        model.best_loss = checkpoint['best_loss']
        try:
            model.best_epochs = checkpoint['best_epochs']
        except KeyError:
            pass
        
        print("Model loaded for resuming training.")
        try:
            showPlot(model.training_losses, model.val_losses, model.savepath)
        except ValueError as e:
            print(e)

    if args.load:
        path = os.path.join(
                os.getcwd(), "recipe_gen", "results", args.model_name, args.load_folder, "best_model")
        try:
            model.load_state_dict(torch.load(path))
        except RuntimeError:
            loaded = torch.load(path,map_location='cuda:0')
            model.load_state_dict(loaded)
            
        print("Model loaded.")

    model.to(args.device)

    for state in model.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)


    if args.train_mode:
        print("Begin training.")
        model.train_process()

    if args.test:
        # Evaluate on whole test dataset
        model.evalOutput()

        # Evaluate on 2 random samples in test dataset and print results
        model.evaluateRandomly(n=2)

        # Evaluate on user input
        sample = {"ingr":"tomato salad beef lemon".split(),
                  "title":"mediteranean salad".split(),
                  "cuisine":"Asian",
                  "id":"user"}
        model.evaluateFromText(sample)

    if args.sample_id:
        model.evalFromId(args.sample_id)
            
    return args

if __name__ == "__main__":
    if sys.modules['__main__'].__file__ == cProfile.__file__:
        import recipe_gen.main  # Imports you again (does *not* use cache or execute as __main__)
        globals().update(vars(recipe_gen.main))  # Replaces current contents with newly imported stuff
        sys.modules['__main__'] = recipe_gen.main  # Ensures pickle lookups on __main__ find matching version
    main() 
