#!/usr/bin/env python
# coding: utf-8

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

from recipe_gen.seq2seq_utils import *
from recipe_gen.network import *
from recipe_gen.hierarchical_network import *

class HierarchicalSeq2seq(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.max_length = args.max_length
        self.max_step = args.max_step

        self.decoder = HierDecoderRNN(args, self.output_size, self.train_dataset.vocab_tokens.idx2word,
                                      self.train_dataset.EOS_token, self.train_dataset.vocab_tokens.word2idx["."])
        # self.decoder_optimizer = optim.Adam(
        #     self.decoder.parameters(), lr=args.learning_rate)
        # self.optim_list[1] = self.decoder_optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

    def initForward(self, input_tensor, pairing=False):
        self.batch_size = batch_size = len(input_tensor)

        # input to Sentence decoder
        decoder_input = torch.zeros(
            (1, batch_size, self.args.hidden_size), device=self.device)
        # (1,batch,hidden)

        sub_decoder_input = torch.tensor(
            [[self.train_dataset.SOS_token]*batch_size], device=self.device)  # input to Word sub_decoder
        # decoder_input final (<max_len,batch)

        decoded_words = [[[] for j in range(self.max_step)] for i in range(
            batch_size)]  # chosen words
        decoder_outputs = torch.zeros(batch_size, self.max_step, self.max_length, len(
            self.train_dataset.vocab_tokens), device=self.device)  # the proba on the words of vocab for each word of each sent

        if pairing:
            decoder_attentions = torch.zeros(self.max_step,
                                             self.max_length, batch_size, self.decoder.pairAttention.pairings.top_k)
            comp_ingrs = torch.zeros(self.max_step, self.max_length, batch_size,
                                     self.decoder.pairAttention.pairings.top_k, dtype=torch.int)
            focused_ingrs = torch.zeros(
                self.max_step, self.max_length, batch_size, dtype=torch.int)

            return decoder_input, sub_decoder_input, decoded_words, decoder_outputs, decoder_attentions, comp_ingrs, focused_ingrs

        else:
            decoder_attentions = torch.zeros(self.max_step,
                                             self.max_length, batch_size, self.max_ingr)

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
            target_tensor = batch["target_instr"].to(
                self.device)  # (batch,max_step,max_length) ?
        except (AttributeError, KeyError):
            target_tensor = None

        decoder_input, sub_decoder_input, decoded_words, sub_decoder_outputs, decoder_attentions = self.initForward(
            input_tensor)

        # encoder_outputs (max_ingr,batch, 2*hidden_size)
        # encoder_hidden (2, batch, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        decoder_hidden = self.encoder_fusion(
            encoder_hidden)  # (2, batch, hidden_size)

        sampling_proba = self.getSamplingProba(iter)

        for cur_step in range(self.max_step):
            decoder_input, decoder_hidden, attn_weights, decoded_words = self.decoder(
                decoder_input, sub_decoder_input, decoder_hidden, sub_decoder_outputs, encoder_outputs, decoded_words, cur_step, target_tensor,
                sampling_proba)  # can remove encoder_outputs ? not used in decoder
            decoder_attentions = self.addAttention(
                cur_step, decoder_attentions, attn_weights)

        return sub_decoder_outputs, decoded_words, {"attentions": decoder_attentions}


class HierarchicalAtt(HierarchicalSeq2seq, Seq2seqAtt):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = HierAttnDecoderRNN(args, self.output_size, self.train_dataset.vocab_tokens.idx2word,
                                          self.train_dataset.EOS_token, self.train_dataset.vocab_tokens.word2idx["."])
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)


class HierarchicalIngrAtt(HierarchicalSeq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = HierIngrAttnDecoderRNN(args, self.output_size, self.train_dataset.vocab_tokens.idx2word,
                                              self.train_dataset.EOS_token, self.train_dataset.vocab_tokens.word2idx["."])
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)


class HierarchicalIngrPairingAtt(HierarchicalSeq2seq, Seq2seq):
    def __init__(self, args):
        super().__init__(args)
        self.decoder = HierPairAttnDecoderRNN(args, self.output_size, self.train_dataset.vocab_tokens.idx2word,
                                              EOS_TOK=self.train_dataset.EOS_token, DOT_TOK=self.train_dataset.vocab_tokens.word2idx["."])
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

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
            target_tensor = batch["target_instr"].to(
                self.device)  # (batch,max_step,max_length) ?
        except (AttributeError, KeyError):
            target_tensor = None

        decoder_input, sub_decoder_input, decoded_words, decoder_outputs, decoder_attentions, comp_ingrs, focused_ingrs = self.initForward(
            input_tensor, pairing=True)  # not same init, at least for decoder_input !!!!

        # encoder_outputs (max_ingr,batch, 2*hidden_size)
        # encoder_hidden (2, batch, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        decoder_hidden = self.encoder_fusion(
            encoder_hidden)  # (2, batch, hidden_size)

        sampling_proba = self.getSamplingProba(iter)

        for cur_step in range(self.max_step):
            decoder_input, decoded_words, decoder_hidden, decoder_attentions = self.forwardDecoderStep(decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, input_tensor, encoder_outputs, decoded_words,
                                                                                                       cur_step, target_tensor, sampling_proba, decoder_attentions, comp_ingrs, focused_ingrs)

        return decoder_outputs, decoded_words, {"attentions": decoder_attentions,
                                                "comp_ingrs": comp_ingrs,
                                                "focused_ingr": focused_ingrs}

    def forwardDecoderStep(self, decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, input_tensor, encoder_outputs, decoded_words, cur_step, target_tensor, sampling_proba, decoder_attentions, comp_ingrs, focused_ingrs):
        decoder_output, decoder_hidden, decoder_attention, decoded_words, comp_ingr, focused_ingr = self.decoder(
            decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, input_tensor,
            self.encoder.embedding, encoder_outputs, decoded_words, cur_step, target_tensor, sampling_proba)

        decoder_attentions = self.addAttention(
            cur_step, decoder_attentions, decoder_attention)
        comp_ingrs[cur_step] = comp_ingr
        focused_ingrs[cur_step] = focused_ingr
        return decoder_output, decoded_words, decoder_hidden, decoder_attentions


class HierarchicalTitlePairing(HierarchicalIngrPairingAtt, Seq2seqTitlePairing):
    def __init__(self, args):
        super().__init__(args)
        self.decoder = HierPairAttnDecoderRNN(args, self.output_size, self.train_dataset.vocab_tokens.idx2word,
                                              EOS_TOK=self.train_dataset.EOS_token, DOT_TOK=self.train_dataset.vocab_tokens.word2idx["."])
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

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
            target_tensor = batch["target_instr"].to(
                self.device)  # (batch,max_step,max_length) ?
        except (AttributeError, KeyError):
            target_tensor = None

        decoder_input, sub_decoder_input, decoded_words, decoder_outputs, decoder_attentions, comp_ingrs, focused_ingrs = self.initForward(
            input_tensor, pairing=True)

        # encoder_outputs (max_ingr,batch, 2*hidden_size)
        # encoder_hidden (2, batch, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        title_tensor = batch["title"].to(self.device)
        title_encoder_outputs, title_encoder_hidden = self.title_encoder.forward_all(
            title_tensor)

        decoder_hidden = torch.cat(
            (encoder_hidden, title_encoder_hidden), dim=2)
        decoder_hidden = self.encoder_fusion(decoder_hidden)

        sampling_proba = self.getSamplingProba(iter)
        for cur_step in range(self.max_step):
            decoder_input, decoded_words, decoder_hidden, decoder_attentions = self.forwardDecoderStep(decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, input_tensor, encoder_outputs, decoded_words,
                                                                                                       cur_step, target_tensor, sampling_proba, decoder_attentions, comp_ingrs, focused_ingrs)

        return decoder_outputs, decoded_words, {"attentions": decoder_attentions,
                                                "comp_ingrs": comp_ingrs,
                                                "focused_ingr": focused_ingrs}


class HierarchicalCuisinePairing(HierarchicalIngrPairingAtt, Seq2seqCuisinePairing):
    def __init__(self, args):
        super().__init__(args)
        self.decoder = HierPairAttnDecoderRNN(args, self.output_size, self.train_dataset.vocab_tokens.idx2word,
                                              EOS_TOK=self.train_dataset.EOS_token, DOT_TOK=self.train_dataset.vocab_tokens.word2idx["."])
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

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
            target_tensor = batch["target_instr"].to(
                self.device)  # (batch,max_step,max_length) ?
        except (AttributeError, KeyError):
            target_tensor = None

        decoder_input, sub_decoder_input, decoded_words, decoder_outputs, decoder_attentions, comp_ingrs, focused_ingrs = self.initForward(
            input_tensor, pairing=True)

        # encoder_outputs (max_ingr,batch, 2*hidden_size)
        # encoder_hidden (2, batch, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        cuisine_tensor = batch["cuisine"].to(self.device)
        cuisine_encoding = self.cuisine_encoder(cuisine_tensor)
        cuisine_encoding = torch.stack(
            [cuisine_encoding] * self.encoder.gru_layers)

        decoder_hidden = torch.cat(
            (encoder_hidden, cuisine_encoding), dim=2)
        decoder_hidden = self.encoder_fusion(decoder_hidden)

        sampling_proba = self.getSamplingProba(iter)
        for cur_step in range(self.max_step):
            decoder_input, decoded_words, decoder_hidden, decoder_attentions = self.forwardDecoderStep(decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, input_tensor, encoder_outputs, decoded_words,
                                                                                                       cur_step, target_tensor, sampling_proba, decoder_attentions, comp_ingrs, focused_ingrs)

        return decoder_outputs, decoded_words, {"attentions": decoder_attentions,
                                                "comp_ingrs": comp_ingrs,
                                                "focused_ingr": focused_ingrs}
