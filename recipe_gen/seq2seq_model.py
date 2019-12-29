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
from recipe_gen.network import *
from recipe_gen.seq2seq_utils import *


class Seq2seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_length = args.max_length
        self.max_ingr = args.max_ingr
        self.train_dataset = RecipesDataset(args)
        self.test_dataset = RecipesDataset(args,train=False)
        self.input_size = input_size = len(self.train_dataset.vocab_ingrs)
        self.output_size = output_size = len(self.train_dataset.vocab_tokens)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=args.batch_size, shuffle=True,
                                                            num_workers=4)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=args.batch_size, shuffle=True,
                                                           num_workers=4)

        self.batch_size = args.batch_size
        self.device = args.device
        self.savepath = args.saving_path
        self.logger = args.logger
        self.train_mode = args.train_mode

        self.encoder = EncoderRNN(args, input_size)
        self.decoder = DecoderRNN(args,output_size)

        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=args.learning_rate)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)

        # Training param
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.learning_rate = args.learning_rate
        self.criterion = nn.NLLLoss(reduction="sum")

        self.paramLogging()

    def paramLogging(self):
        for k,v in self.args.defaults.items():
            try:
                if getattr(self.args,k)!=v and v is not None:
                    self.logger.info("{} = {}".format(k,getattr(self,k)))
            except AttributeError:
                continue

    def addAttention(self, di, decoder_attentions, cur_attention):
        if cur_attention is not None:
            decoder_attentions[di] = cur_attention.data
        return decoder_attentions
    
    def samplek(self,decoder_output,decoded_words):
        topv, topi = decoder_output.topk(self.args.topk)
        distrib= torch.distributions.categorical.Categorical(logits=topv)
        chosen_id = torch.zeros(decoder_output.shape[0],dtype=torch.long,device=self.device)
        for batch_id,idx in enumerate(distrib.sample()):
            chosen_id[batch_id] = topi[batch_id,idx]
            decoded_words[batch_id].append(
                self.train_dataset.vocab_tokens.idx2word[chosen_id[batch_id].item()])
        return chosen_id

    def initForward(self,input_tensor,pairing=False):
        self.batch_size = len(input_tensor)
        decoder_input = torch.tensor(
            [[self.train_dataset.SOS_token]*self.batch_size], device=self.device)
        # decoder_input final (<max_len,batch)

        decoded_words = [[] for i in range(self.batch_size)]
        decoder_outputs = torch.zeros(self.batch_size, self.max_length, len(
            self.train_dataset.vocab_tokens), device=self.device)

        if pairing:
            decoder_attentions = torch.zeros(
                self.max_length, self.batch_size, self.decoder.pairAttention.pairings.top_k)
        else:
            decoder_attentions = torch.zeros(
                self.max_length, self.batch_size, self.max_ingr)
    
        return decoder_input,decoded_words,decoder_outputs,decoder_attentions
    

    def forward(self, batch):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """
        def forwardDecoderStep(decoder_input, decoder_hidden, 
                    encoder_outputs, di, decoder_attentions, decoder_outputs, decoded_words):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[:, di] = decoder_output
            decoder_attentions = self.addAttention(
                di, decoder_attentions, decoder_attention)
            topi = self.samplek(decoder_output,decoded_words)
            return decoder_hidden, decoder_attentions, topi


        input_tensor = batch["ingr"].to(self.device)
        target_tensor = batch["target_instr"].to(self.device)
        
        decoder_input,decoded_words,decoder_outputs,decoder_attentions = self.initForward(input_tensor)

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        # encoder_outputs (max_ingr,hidden_size, batch)
        # encoder_hidden (1, batch, hidden_size)

        decoder_hidden = encoder_hidden

        if self.training:
            use_teacher_forcing = True if random.random(
            ) < self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        
        # TODO: scheduled sampling
        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_hidden, decoder_attentions, topi = forwardDecoderStep(decoder_input, decoder_hidden, 
                    encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words)

                decoder_input = target_tensor[:, di].view(1, -1)

        else:
            for di in range(self.max_length):
                decoder_hidden, decoder_attentions, topi = forwardDecoderStep(decoder_input, decoder_hidden, 
                    encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words)
                    
                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[:, 0]
                if len(idx_end) == self.batch_size:
                    break

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]

    def train_iter(self, batch):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        target_length = batch["target_length"] 
        target_tensor = batch["target_instr"].to(self.device)

        decoded_outputs, decoded_words, _ = self.forward(batch)
        # aligned_outputs = flattenSequence(decoded_outputs, target_length)
        # aligned_target = flattenSequence(target_tensor, target_length)
        aligned_outputs = decoded_outputs.view(self.batch_size*self.max_length,-1)
        aligned_target = target_tensor.view(self.batch_size*self.max_length)
        loss = self.criterion(aligned_outputs, aligned_target)/target_length.shape[0]
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item(), decoded_words

    def train_process(self):
        self.train()
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        best_loss=math.inf

        for ep in range(1,self.args.epoch+1):
            for iter, batch in enumerate(self.train_dataloader, start=1):
                if iter == self.args.n_iters:
                    break

                loss, decoded_words = self.train_iter(batch)
                print_loss_total += loss
                plot_loss_total += loss

                if iter % self.args.print_step == 0:
                    print_loss_avg = print_loss_total / self.args.print_step
                    print_loss_total = 0
                    self.logger.info('Epoch {} {} ({} {}%) loss={}'.format(ep,timeSince(start, iter / self.args.n_iters),iter,int(iter / self.args.n_iters * 100),print_loss_avg))
                    print(" ".join(decoded_words[0]))

                    torch.save(self.state_dict(), os.path.join(
                            self.savepath, "model_{}_{}".format(datetime.now().strftime('%m-%d-%H-%M'),iter)))
                    if print_loss_avg<best_loss:
                        torch.save(self.state_dict(), os.path.join(
                            self.savepath, "best_model"))

    def evaluate(self, sentence, target=None):
        self.eval()
        with torch.no_grad():
            input_tensor, _ = self.train_dataset.tensorFromSentence(
                self.train_dataset.vocab_ingrs, sentence)
            input_tensor = input_tensor.view(1, -1).to(self.device)
            if target is not None:
                target_tensor, _ = self.train_dataset.tensorFromSentence(
                    self.train_dataset.vocab_tokens, target, instructions=True)
                target_tensor = target_tensor.view(1, -1).to(self.device)
            else:
                target_tensor = None
            return self.forward(input_tensor, target_tensor)

    def evaluateRandomly(self, n=10):
        for i in range(n):
            pair = random.choice(self.test_dataset.pairs)
            loss, output_words, _ = self.evaluate(pair[0], pair[1])
            output_sentence = ' '.join(output_words[0])

            self.logger.info("Input: "+" ".join([ingr.name for ingr in pair[0]]))
            self.logger.info("Target: "+[" ".join(instr) for instr in pair[1]])
            self.logger.info("Generated: "+output_sentence)

    def evalProcess(self):
        start = time.time()
        plot_losses = []
        print_loss_total = 0

        for iter, batch in enumerate(self.test_dataloader, start=1):
            # split in train_iter? give directly batch to train_iter ?
            input_tensor = batch["ingr"].to(self.device)
            target_tensor = batch["target_instr"].to(self.device)
            target_length = batch["target_length"]

            loss = self.train_iter(input_tensor, target_tensor, target_length)
            print_loss_total += loss

        print_loss_avg = print_loss_total / self.args.print_step
        self.logger.info("Eval loss = {.4f}".format(print_loss_avg))

class Seq2seqAtt(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = AttnDecoderRNN(args, self.output_size)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)

    def evaluateAndShowAttention(self, input_sentence):
        loss, output_words, attentions = self.evaluate(input_sentence)
        self.logger.info('input = '+ input_sentence)
        self.logger.info('output = '+ ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)


class Seq2seqIngrAtt(Seq2seqAtt):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = IngrAttnDecoderRNN(args, self.output_size)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)


class Seq2seqIngrPairingAtt(Seq2seqAtt):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = PairAttnDecoderRNN(args, self.output_size, unk_token=self.train_dataset.UNK_token)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)

    def forward(self, batch):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """
        def forwardDecoderStep(decoder_input, decoder_hidden, 
                    encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, self.encoder.embedding, input_tensor)

            decoder_attentions = self.addAttention(di, decoder_attentions, decoder_attention)
            decoder_outputs[:, di, :] = decoder_output
            topi = self.samplek(decoder_output,decoded_words)
            return decoder_hidden, decoder_attentions, topi

        input_tensor = batch["ingr"].to(self.device)
        target_tensor = batch["target_instr"].to(self.device)

        decoder_input,decoded_words,decoder_outputs,decoder_attentions = self.initForward(input_tensor,pairing=True)

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        # encoder_outputs (hidden_size, batch) or (max_ingr,hidden) ??
        # encoder_hidden (1, batch, hidden_size)


        decoder_hidden = encoder_hidden

        if self.training:
            use_teacher_forcing = True if random.random(
            ) < self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_hidden, decoder_attentions,topi = forwardDecoderStep(decoder_input, decoder_hidden, 
                    encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words)

                decoder_input = target_tensor[:, di].view(1,-1)

        else:
            for di in range(self.max_length):
                decoder_hidden, decoder_attentions,topi = forwardDecoderStep(decoder_input, decoder_hidden, 
                    encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words)

                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[:, 0]
                if len(idx_end) == self.batch_size:
                    break

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]



class Seq2seqTitlePairing(Seq2seqAtt):
    def __init__(self, args):
        super().__init__(args)
        self.title_encoder = EncoderRNN(args,self.input_size)
        self.encoder_optimizer = optim.Adam(
            self.title_encoder.parameters(), lr=args.learning_rate)

        self.decoder = PairAttnDecoderRNN(args, self.output_size, unk_token=self.train_dataset.UNK_token)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)

    def forward(self, batch):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """
        input_tensor = batch["ingr"].to(self.device)
        target_tensor = batch["target_instr"].to(self.device)
        title_tensor = batch["title"].to(self.device)

        decoder_input,decoded_words,decoder_outputs,decoder_attentions = self.initForward(input_tensor,pairing=True)

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        # encoder_outputs (max_ingr,hidden_size, batch)
        # encoder_hidden (1,hidden_size, batch)

        title_encoder_outputs, title_encoder_hidden = self.title_encoder.forward_all(
            title_tensor)

        decoder_hidden = torch.cat((encoder_hidden,title_encoder_hidden),dim=2)

        if self.training:
            use_teacher_forcing = True if random.random(
            ) < self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, self.encoder.embedding, input_tensor)

                decoder_attentions = self.addAttention(di, decoder_attentions, decoder_attention)
                decoder_outputs[:, di, :] = decoder_output
                decoder_input = target_tensor[:, di].view(1,-1)

                self.samplek(decoder_output,decoded_words)

        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, self.encoder.embedding, input_tensor)

                decoder_attentions = self.addAttention(di, decoder_attentions, decoder_attention)
                decoder_outputs[:, di, :] = decoder_output

                topi = self.samplek(decoder_output,decoded_words)
                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[:, 0]
                if len(idx_end) == self.batch_size:
                    break

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]
