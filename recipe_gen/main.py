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

# Run settings
argparser.add_argument('--model-name', type=str, choices=[
                       'Seq2seq','Seq2seqAtt','Seq2seqIngrAtt','Seq2seqIngrPairingAtt',
                       'Seq2seqTitlePairing','Seq2seqCuisinePairing','Seq2seqTrans',
                       'HierarchicalSeq2seq','HierarchicalSeq2seqAtt','HierarchicalSeq2seqIngrAtt',
                       'HierarchicalSeq2seqIngrPairingAtt'],
                       default="Seq2seqIngrPairingAtt",
                       help='Model name for saving/loading')
argparser.add_argument('--print-step', type=int, default=50,
                       help='Display steps')
argparser.add_argument('--validation-step', type=int, default=1, #XXX: utility ?
                       help='Number of random search validation')
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
argparser.add_argument('--decay_factor', type=float, default=500.,
                        help='Speed of increasing the probability of sampling from model. Default: 500.')
argparser.add_argument('--dropout', type=float, default=0.1)
argparser.add_argument('--topk', type=int, default=3)
argparser.add_argument('--scheduled-sampling', type='bool', nargs='?',
                        const=True, default=True,
                       help='Uses scheduled-sampling')

# Model config
argparser.add_argument('--hidden-size', type=int, default=128)
argparser.add_argument('--max-ingr', type=int, default=10)
argparser.add_argument('--max-length', type=int, default=100) 
argparser.add_argument('--max-step', type=int, default=10) #for hierarchical, then max-length <30 is better, per step

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

    try:
        model_class=getattr(importlib.import_module("recipe_gen.seq2seq_model"), args.model_name)
    except AttributeError:
        model_class=getattr(importlib.import_module("recipe_gen.hierarchical_model"), args.model_name)

    model = model_class(args)

    if args.resume:
        checkpoint = torch.load(os.path.join(
            os.getcwd(), "recipe_gen", "results",args.model_name,args.load_folder))
        model.load_state_dict(checkpoint['model_state_dict'])
        [optim.load_state_dict(checkpoint['optimizer_state_dict'][i]) for i,optim in enumerate(model.optim_list)]
        args.begin_epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        print("Model loaded for resuming training.")

    if args.load and not args.resume:
        model.load_state_dict(torch.load(os.path.join(
            os.getcwd(), "recipe_gen", "results",args.model_name,args.load_folder, "best_model"))) 
            #for consistency, don't add best_model, ask user to add it ?
        print("Model loaded.")
    
    model.to(args.device)

    for optim in model.optim_list:
        for state in optim.state.values():
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
        _, output_words, attentions = model.evaluateFromText(
            "tomato salad beef lemon".split(),title="mediteranean salad".split())
        try:
            print(' '.join(output_words[0]))
        except TypeError:
            print([" ".join(sent) for sent in output_words[0]])

        try:
            plt.matshow(attentions[:, 0, :].numpy())
        except (TypeError, AttributeError):
            print("No attention to show.")



if __name__ == "__main__":
    main()
