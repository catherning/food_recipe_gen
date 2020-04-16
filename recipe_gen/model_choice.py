import os
import sys
import torch
import math
import logging
import importlib
import argparse
sys.path.insert(0,os.getcwd())

from recipe_gen.main import argparser, getDefaultArgs, init_logging, init_seed
from recipe_gen.pairing_utils import PairingData

args = getDefaultArgs(argparser)
args.logger = LOGGER = logging.getLogger()
init_logging(args)
init_seed(args)

def loadModel(model,mod):
    checkpoint = torch.load(os.path.join(path, mod))
    try:
      del checkpoint['model_state_dict']["decoder.attention.attn_combine.weight"]
      del checkpoint['model_state_dict']["decoder.attention.attn_combine.bias"]
    except KeyError:
      pass
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    print("Model loaded {}".format(mod))


try:
    model_class = getattr(importlib.import_module(
        "recipe_gen.seq2seq_model"), args.model_name)
except AttributeError:
    model_class = getattr(importlib.import_module(
        "recipe_gen.hierarchical_model"), args.model_name)
    
model = model_class(args)


path = os.path.join(
    os.getcwd(), "recipe_gen", "results", args.model_name,args.load_folder)
model_list = filter(lambda f: f[0]=="t",os.listdir(path))
    
best_loss = math.inf          
for mod in model_list:

    loadModel(model,mod)
    
    eval_loss = model.evalProcess()
    if eval_loss < best_loss:
        best_model = mod
        best_loss = eval_loss

LOGGER.info("Best model = {}".format(best_model))
LOGGER.info("Eval loss = {}".format(best_loss))
loadModel(model,best_model)
    
torch.save(model.state_dict(), os.path.join(
                args.saving_path, "best_model"))
    
