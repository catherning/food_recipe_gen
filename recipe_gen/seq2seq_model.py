#!/usr/bin/env python
# coding: utf-8

from recipe_gen.seq2seq_utils import *
from recipe_gen.network import *
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


class Seq2seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_length = args.max_length
        self.max_ingr = args.max_ingr
        self.train_dataset = RecipesDataset(args)
        self.test_dataset = RecipesDataset(args, train=False)
        self.input_size = input_size = len(self.train_dataset.vocab_ingrs)
        self.output_size = output_size = len(self.train_dataset.vocab_tokens)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=args.batch_size, shuffle=True,
                                                            num_workers=0)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=args.batch_size, shuffle=False,
                                                           num_workers=0)

        self.batch_size = args.batch_size
        self.device = args.device
        self.savepath = args.saving_path
        self.logger = args.logger
        self.train_mode = args.train_mode

        self.encoder = EncoderRNN(args, input_size)
        self.decoder = DecoderRNN(args, output_size)

        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=args.learning_rate)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list = [self.encoder_optimizer, self.decoder_optimizer]

        # Training param
        self.decay_factor = args.decay_factor
        self.learning_rate = args.learning_rate
        self.criterion = nn.NLLLoss(reduction="sum")

        self.paramLogging()

    def paramLogging(self):
        for k, v in self.args.defaults.items():
            try:
                if getattr(self.args, k) != v and v is not None:
                    self.logger.info("{} = {}".format(
                        k, getattr(self.args, k)))
            except AttributeError:
                continue

    def addAttention(self, di, decoder_attentions, cur_attention):
        if cur_attention is not None:
            decoder_attentions[di] = cur_attention.data
        return decoder_attentions

    def samplek(self, decoder_output, decoded_words):
        # TODO: change for hierarchical
        topv, topi = decoder_output.topk(self.args.topk)
        distrib = torch.distributions.categorical.Categorical(logits=topv)
        chosen_id = torch.zeros(
            decoder_output.shape[0], dtype=torch.long, device=self.device)
        for batch_id, idx in enumerate(distrib.sample()):
            chosen_id[batch_id] = topi[batch_id, idx]
            decoded_words[batch_id].append(
                self.train_dataset.vocab_tokens.idx2word[chosen_id[batch_id].item()])
        return chosen_id

    def initForward(self, input_tensor, pairing=False):
        # XXX: should be able not to reassign if do view with correct hidden size instead
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

        return decoder_input, decoded_words, decoder_outputs, decoder_attentions

    def forwardDecoderStep(self, decoder_input, decoder_hidden,
                           encoder_outputs, di, decoder_attentions, decoder_outputs, decoded_words):
        decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs)  # can remove encoder_outputs ? not used in decoder
        decoder_outputs[:, di] = decoder_output
        decoder_attentions = self.addAttention(
            di, decoder_attentions, decoder_attention)
        topi = self.samplek(decoder_output, decoded_words)
        return decoder_attentions, decoder_hidden, topi

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
        target_tensor = batch["target_instr"].to(self.device)
        decoder_input, decoded_words, decoder_outputs, decoder_attentions = self.initForward(
            input_tensor)

        # encoder_outputs (max_ingr,batch, 2*hidden_size)
        # encoder_hidden (2, batch, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        decoder_hidden = encoder_hidden  # (2, batch, hidden_size)

        sampling_proba = 1-inverse_sigmoid_decay(
            self.decay_factor, iter) if self.training else 1

        for di in range(self.max_length):
            decoder_attentions, decoder_hidden, topi = self.forwardDecoderStep(decoder_input, decoder_hidden,
                                                               encoder_outputs, di, decoder_attentions, decoder_outputs, decoded_words)

            if random.random() < sampling_proba:
                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break
                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input
            else:
                decoder_input = target_tensor[:, di].view(1, -1)

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]

    def train_iter(self, batch, iter):
        for optim in self.optim_list:
            optim.zero_grad()

        target_length = batch["target_length"]
        target_tensor = batch["target_instr"].to(self.device)

        decoded_outputs, decoded_words, _ = self.forward(batch, iter=iter)
        #TODO: change flattenSeq if hierarchical
        aligned_outputs = flattenSequence(decoded_outputs, target_length) 
        aligned_target = flattenSequence(target_tensor, target_length)
        # aligned_outputs = decoded_outputs.view(
        #     self.batch_size*self.max_length, -1)
        # aligned_target = target_tensor.view(self.batch_size*self.max_length)
        loss = self.criterion(
            aligned_outputs, aligned_target)/self.batch_size

        if self.training:
            loss.backward()

            for optim in self.optim_list:
                optim.step()

        return loss.item(), decoded_words

    def train_process(self):
        self.train()
        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0
        best_loss = math.inf

        lmbda = lambda epoch: 0.95
        scheduler_list = [torch.optim.lr_scheduler.MultiplicativeLR(optim,lr_lambda=lmbda) for optim in self.optim_list]
        for ep in range(self.args.begin_epoch, self.args.epoch+1):
            for iter, batch in enumerate(self.train_dataloader, start=1):
                if iter == self.args.n_iters:
                    break

                loss, decoded_words = self.train_iter(batch, iter)
                print_loss_total += loss
                plot_loss_total += loss

                if iter % max(self.args.print_step,self.args.n_iters//10) == 0:
                    print_loss_avg = print_loss_total / max(self.args.print_step,self.args.n_iters//10)
                    print_loss_total = 0
                    self.logger.info('Epoch {} {} ({} {}%) loss={}'.format(ep, timeSince(
                        start, iter / self.args.n_iters), iter, int(iter / self.args.n_iters * 100), print_loss_avg))
                    
                    try:
                        print(" ".join(decoded_words[0]))
                        print(" ".join([self.train_dataset.vocab_tokens.idx2word[word.item()]  for word in batch["target_instr"][0] if word.item()!=0]))
                    except TypeError:
                        print([" ".join(sent) for sent in decoded_words[0]])
                        print([" ".join([self.train_dataset.vocab_tokens.idx2word[word.item()] for word in sent if word.item()!=0]) for sent in batch["target_instr"][0]])


                    torch.save({
                        'epoch': ep,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': [optim.state_dict() for optim in self.optim_list],
                        'loss': loss,
                        }, os.path.join(self.savepath,"train_model_{}_{}.tar".format(datetime.now().strftime('%m-%d-%H-%M'), iter)))

                    if print_loss_avg < best_loss:
                        print("Best model so far, saving it.")
                        torch.save(self.state_dict(), os.path.join(
                            self.savepath, "best_model"))
                        best_loss = print_loss_avg

            if ep%(max(5,self.args.epoch//10))==0: #eval ten times or every 5 times
                self.evalProcess()
            
            for scheduler in scheduler_list:
                scheduler.step()

    def evaluateFromText(self, ingr_list, target=None, title=None):
        self.eval()
        with torch.no_grad():
            input_tensor, _ = self.train_dataset.tensorFromSentence(
                self.train_dataset.vocab_ingrs, ingr_list)
            input_tensor = input_tensor.view(1, -1).to(self.device)

            if target is not None:
                target_tensor, _ = self.train_dataset.tensorFromSentence(
                    self.train_dataset.vocab_tokens, target, instructions=True)
                target_tensor = target_tensor.view(1, -1).to(self.device)
            else:
                target_tensor = None

            if title is not None:
                title_tensor, _ = self.train_dataset.tensorFromSentence(
                    self.train_dataset.vocab_tokens, title)
                title_tensor = title_tensor.view(1, -1).to(self.device)
            else:
                title_tensor = None

            batch = {"ingr": input_tensor,
                     "target_instr": target_tensor, "title": title_tensor}
            return self.forward(batch)

    def evaluateRandomly(self, n=10):
        self.eval()
        for i in range(n):
            sample = random.choice(self.test_dataset.data)
            sample["ingr"] = sample["ingr"].unsqueeze(0)
            loss, output_words, _ = self.forward(sample)
            try:
                output_sentence = ' '.join(output_words[0])
            except TypeError:
                output_sentence= "|".join([' '.join(sent) for sent in output_words[0]])


            self.logger.info(
                "Input: "+" ".join([self.train_dataset.vocab_ingrs.idx2word[ingr.item()][0] for ingr in sample["ingr"][0]]))
            self.logger.info(
                "Target: "+str([" ".join([self.train_dataset.vocab_tokens.idx2word[word.item()] for word in instr if word.item()!=0]) for instr in sample["target_instr"]]))
            self.logger.info("Generated: "+output_sentence)

    def evalProcess(self):
        self.eval()
        start = time.time()
        plot_losses = []
        print_loss_total = 0

        for iter, batch in enumerate(self.test_dataloader, start=1):
            loss, _ = self.train_iter(batch, iter)
            print_loss_total += loss

            if iter % self.args.print_step == 0:
                print("Eval Current loss = {}".format(print_loss_total/iter))

        print_loss_avg = print_loss_total / iter
        self.logger.info("Eval loss = {}".format(print_loss_avg))
    
    def evalOutput(self):
        self.eval()
        start = time.time()
        plot_losses = []
        print_loss_total = 0

        for iter, batch in enumerate(self.test_dataloader, start=1):
            loss, output_words = self.train_iter(batch, iter)

            for i,ex in enumerate(output_words):
                try:
                    self.logger.info(batch["id"][i]+" "+' '.join(ex))
                except TypeError:
                    self.logger.info(batch["id"][i]+" "+"|".join([' '.join(sent) for sent in ex ]))

            print_loss_total += loss

            if iter % self.args.print_step == 0:
                print("Current loss = {}".format(print_loss_total/iter))

        print_loss_avg = print_loss_total / iter
        self.logger.info("Eval loss = {}".format(print_loss_avg))


class Seq2seqAtt(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = AttnDecoderRNN(args, self.output_size)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list[1] = self.decoder_optimizer

    def evaluateAndShowAttention(self, input_sentence):
        loss, output_words, attentions = self.evaluateFromText(input_sentence)
        self.logger.info('input = ' + input_sentence)
        self.logger.info('output = ' + ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)


class Seq2seqTrans(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoderLayer = nn.TransformerDecoderLayer(d_model=args.hidden_size, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoderLayer, 4)
        
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list[1] = self.decoder_optimizer


class Seq2seqIngrAtt(Seq2seqAtt):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = IngrAttnDecoderRNN(args, self.output_size)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list[1] = self.decoder_optimizer
        

class Seq2seqIngrPairingAtt(Seq2seqAtt):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = PairAttnDecoderRNN(
            args, self.output_size, unk_token=self.train_dataset.UNK_token)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list[1] = self.decoder_optimizer

    def forwardDecoderStep(self, decoder_input, decoder_hidden,
                           encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words):
        decoder_output, decoder_hidden, decoder_attention = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs, self.encoder.embedding, input_tensor)

        decoder_attentions = self.addAttention(
            di, decoder_attentions, decoder_attention)
        decoder_outputs[:, di] = decoder_output
        topi = self.samplek(decoder_output, decoded_words)
        return decoder_attentions, decoder_hidden, topi

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
        target_tensor = batch["target_instr"].to(self.device)

        decoder_input, decoded_words, decoder_outputs, decoder_attentions = self.initForward(
            input_tensor, pairing=True)

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        # encoder_outputs (hidden_size, batch) or (max_ingr,hidden) ??
        # encoder_hidden (1, batch, hidden_size)

        decoder_hidden = encoder_hidden

        sampling_proba = 1-inverse_sigmoid_decay(
            self.decay_factor, iter) if self.training else 1

        for di in range(self.max_length):
            decoder_attentions, decoder_hidden, topi = self.forwardDecoderStep(decoder_input, decoder_hidden,
                                                               encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words)

            if random.random() < sampling_proba:
                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input
            else:
                decoder_input = target_tensor[:, di].view(1, -1)

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]


class Seq2seqTitlePairing(Seq2seqIngrPairingAtt):
    def __init__(self, args):
        super().__init__(args)
        self.title_encoder = EncoderRNN(args, self.output_size)
        # output because tok of  title are in vocab_toks
        self.title_optimizer = optim.Adam(
            self.title_encoder.parameters(), lr=args.learning_rate)
        self.optim_list.append(self.title_optimizer)

        self.encoder_fusion = nn.Linear(2*args.hidden_size,args.hidden_size)
        self.fusion_optim = optim.Adam(self.encoder_fusion,lr=args.learning_rate)
        self.optim_list.append(self.fusion_optim)

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
        decoder_input, decoded_words, decoder_outputs, decoder_attentions = self.initForward(
            input_tensor, pairing=True)

        try:
            target_tensor = batch["target_instr"].to(self.device)
        except AttributeError:
            Warning("Evaluation mode: only taking ingredient list as input")

        title_tensor = batch["title"].to(self.device)

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        # encoder_outputs (max_ingr,hidden_size, batch)
        # encoder_hidden (1,hidden_size, batch)

        title_encoder_outputs, title_encoder_hidden = self.title_encoder.forward_all(
            title_tensor)

        decoder_hidden = torch.cat(
            (encoder_hidden, title_encoder_hidden), dim=2)
        
        decoder_hidden = self.encoder_fusion(decoder_hidden)

        sampling_proba = 1-inverse_sigmoid_decay(
            self.decay_factor, iter) if self.training else 1

        for di in range(self.max_length):
            decoder_attentions, decoder_hidden, topi = self.forwardDecoderStep(
                decoder_input, decoder_hidden, encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words)

            if random.random() < sampling_proba:
                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input
            else:
                decoder_input = target_tensor[:, di].view(1, -1)

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]


class Seq2seqCuisinePairing(Seq2seqIngrPairingAtt):
    def __init__(self, args):
        super().__init__(args)
        self.cuisine_encoder = EncoderRNN(
            args, len(self.train_dataset.vocab_cuisine))
        self.cuisine_optimizer = optim.Adam(
            self.cuisine_encoder.parameters(), lr=args.learning_rate)
        self.optim_list.append(self.cuisine_optimizer)

        self.encoder_fusion = nn.Linear(2*args.hidden_size,args.hidden_size)
        self.fusion_optim = optim.Adam(self.encoder_fusion,lr=args.learning_rate)
        self.optim_list.append(self.fusion_optim)

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
        decoder_input, decoded_words, decoder_outputs, decoder_attentions = self.initForward(
            input_tensor, pairing=True)

        try:
            target_tensor = batch["target_instr"].to(self.device)
        except AttributeError:
            Warning("Evaluation mode: only taking ingredient list as input")

        # XXX: if only change, that, can just rename both title and cuisine to a fusion
        title_tensor = batch["cuisine"].to(self.device)

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        # encoder_outputs (max_ingr,hidden_size, batch)
        # encoder_hidden (1,hidden_size, batch)

        title_encoder_outputs, title_encoder_hidden = self.title_encoder.forward_all(
            title_tensor)

        decoder_hidden = torch.cat(
            (encoder_hidden, title_encoder_hidden), dim=2)

        decoder_hidden = self.encoder_fusion(decoder_hidden)

        sampling_proba = 1-inverse_sigmoid_decay(
            self.decay_factor, iter) if self.training else 1

        for di in range(self.max_length):
            decoder_attentions, decoder_hidden, topi = self.forwardDecoderStep(
                decoder_input, decoder_hidden, encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words)

            if random.random() < sampling_proba:
                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input
            else:
                decoder_input = target_tensor[:, di].view(1, -1)

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]
