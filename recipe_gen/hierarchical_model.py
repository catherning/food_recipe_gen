#!/usr/bin/env python
# coding: utf-8

from recipe_gen.seq2seq_utils import *
from recipe_gen.network import *
from recipe_gen.hierarchical_network import *
import math
import os
import pickle
import random
import re
import sys
import time
import unicodedata
from datetime import datetime
from functools import reduce
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

sys.path.insert(0, os.getcwd())

class HierarchicalSeq2seq(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.max_length = args.max_length
        self.max_step = args.max_step

        self.decoder = HierDecoderRNN(args, self.output_size,self.train_dataset.vocab_tokens.idx2word,self.train_dataset.EOS_token,self.train_dataset.vocab_tokens.word2idx["."])
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list[1] = self.decoder_optimizer

    def initForward(self, input_tensor, pairing=False):
        self.batch_size = len(input_tensor)
        
        sub_decoder_input = torch.tensor(
            [[self.train_dataset.SOS_token]*self.batch_size], device=self.device) #input to sub_decoder
        # decoder_input final (<max_len,batch)

        decoder_input = torch.tensor(
            [[self.train_dataset.SOS_token]*self.batch_size], device=self.device) #input to sub_decoder

        decoded_words = [[[] for j in range(self.max_step)] for i in range(self.batch_size)] # chosen words
        decoder_outputs = torch.zeros(self.batch_size, self.max_step,self.max_length, len(
            self.train_dataset.vocab_tokens), device=self.device) #the proba on the words of vocab for each word 

        if pairing:
            decoder_attentions = torch.zeros(self.max_step,
                self.max_length, self.batch_size, self.decoder.pairAttention.pairings.top_k)
        else:
            decoder_attentions = torch.zeros(self.max_step,
                self.max_length, self.batch_size, self.max_ingr)

        return decoder_input, sub_decoder_input, decoded_words, decoder_outputs, decoder_attentions

    def forward(self, batch, iter=iter):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """
        input_tensor = batch["ingr"].to(self.device)
        try:
            target_tensor = batch["target_instr"].to(self.device) #(batch,max_step,max_length) ?
        except AttributeError:
            target_tensor = None

        decoder_input, sub_decoder_input, decoded_words, decoder_outputs,decoder_attentions = self.initForward(
            input_tensor) # not same init, at least for decoder_input !!!!

        # encoder_outputs (max_ingr,batch, 2*hidden_size)
        # encoder_hidden (2, batch, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        decoder_hidden = encoder_hidden  # (2, batch, hidden_size)

        sampling_proba = 1-inverse_sigmoid_decay(
            self.decay_factor, iter) if self.training else 1

        for cur_step in range (self.max_step):
            decoder_output, decoder_hidden, attn_weights, decoded_words = self.decoder(
                    decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, encoder_outputs,decoded_words,cur_step,target_tensor,
                    sampling_proba)  # can remove encoder_outputs ? not used in decoder
            decoder_attentions = self.addAttention(
                cur_step, decoder_attentions, attn_weights)

        return decoder_outputs, decoded_words, decoder_attentions

class HierarchicalSeq2seqAtt(HierarchicalSeq2seq,Seq2seqAtt):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = HierAttnDecoderRNN(args, self.output_size,self.train_dataset.vocab_tokens.idx2word,self.train_dataset.EOS_token,self.train_dataset.vocab_tokens.word2idx["."])
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list[1] = self.decoder_optimizer
    

class HierarchicalSeq2seqIngrAtt(HierarchicalSeq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = HierIngrAttnDecoderRNN(args, self.output_size,self.train_dataset.vocab_tokens.idx2word,self.train_dataset.EOS_token,self.train_dataset.vocab_tokens.word2idx["."])
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list[1] = self.decoder_optimizer

  
class HierarchicalSeq2seqIngrPairingAtt(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.max_length = args.max_length
        self.max_step = args.max_step

        self.decoder = HierDecoderRNN(args, self.output_size,self.train_dataset.vocab_tokens.idx2word,self.train_dataset.EOS_token,self.train_dataset.vocab_tokens.word2idx["."])
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list[1] = self.decoder_optimizer

    def forward(self, batch, iter=iter):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """
        input_tensor = batch["ingr"].to(self.device)
        try:
            target_tensor = batch["target_instr"].to(self.device) #(batch,max_step,max_length) ?
        except AttributeError:
            target_tensor = None

        decoder_input, sub_decoder_input, decoded_words, decoder_outputs = self.initForward(
            input_tensor) # not same init, at least for decoder_input !!!!

        # encoder_outputs (max_ingr,batch, 2*hidden_size)
        # encoder_hidden (2, batch, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        decoder_hidden = encoder_hidden  # (2, batch, hidden_size)

        sampling_proba = 1-inverse_sigmoid_decay(
            self.decay_factor, iter) if self.training else 1

        for cur_step in range (self.max_step):
            decoder_output, decoder_hidden, _, decoded_words = self.decoder(
                    decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, input_tensor, encoder_outputs,decoded_words,cur_step,target_tensor,
                    sampling_proba)  # can remove encoder_outputs ? not used in decoder

        return decoder_outputs, decoded_words, None # decoder_attentions[:di + 1]
