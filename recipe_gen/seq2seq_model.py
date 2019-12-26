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
        self.max_length = args.max_length
        self.max_ingr = args.max_ingr
        self.train_dataset = RecipesDataset(args)
        self.test_dataset = RecipesDataset(args,train=False)
        self.args = args
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
        self.savepath = os.path.join(args.saving_path,self.__class__.__name__,datetime.now().strftime('%m-%d-%H-%M'))
        try:
            os.makedirs(self.savepath)
        except FileExistsError:
            pass

        self.encoder = EncoderRNN(args, input_size)
        self.decoder = DecoderRNN(args,output_size)

        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=args.learning_rate)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)

        # Training param
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.learning_rate = args.learning_rate
        self.criterion = nn.NLLLoss()

    def addAttention(self, di, decoder_attentions, cur_attention):
        if cur_attention is not None:
            decoder_attentions[di] = cur_attention.data
        return decoder_attentions

    def forward(self, input_tensor, target_tensor):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """
        self.batch_size = len(input_tensor)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        # encoder_outputs (max_ingr, batch,hidden)
        # encoder_hidden (1, batch, hidden_size)

        decoder_input = torch.tensor(
            [[self.data.SOS_token]*self.batch_size], device=self.device)
        decoder_hidden = encoder_hidden
        # decoder_input final (<max_len,batch)

        if self.training:
            use_teacher_forcing = True if random.random(
            ) < self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        decoded_words = [[] for i in range(self.batch_size)]
        decoder_outputs = torch.zeros(self.batch_size, self.max_length, len(
            self.data.vocab_tokens), device=self.device)
        decoder_attentions = torch.zeros(
            self.max_length, self.batch_size, self.max_ingr)

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[:, di] = decoder_output
                decoder_input = target_tensor[:, di].view(1, -1)
                decoder_attentions = self.addAttention(
                    di, decoder_attentions, decoder_attention)

                topv, topi = decoder_output.topk(1)
                for batch_id, word_id in enumerate(topi):
                    decoded_words[batch_id].append(
                        self.data.vocab_tokens.idx2word[word_id.item()])

        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_outputs[:, di, :] = decoder_output
                decoder_attentions = self.addAttention(
                    di, decoder_attentions, decoder_attention)

                idx_end = (topi == self.data.EOS_token).nonzero()[:, 0]
                if len(idx_end) == self.batch_size:
                    break

                for batch_id, word_id in enumerate(topi):
                    decoded_words[batch_id].append(
                        self.data.vocab_tokens.idx2word[word_id.item()])

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]

    def train_iter(self, input_tensor, target_tensor, target_length):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        decoded_outputs, decoded_words, _ = self.forward(
            input_tensor, target_tensor)
        # aligned_outputs = flattenSequence(decoded_outputs, target_length)
        # aligned_target = flattenSequence(target_tensor, target_length)
        aligned_outputs = decoded_outputs.view(self.batch_size*self.max_length,-1)
        aligned_target = target_tensor.view(self.batch_size*self.max_length)
        loss = self.criterion(aligned_outputs, aligned_target)
        # XXX: should not do mean in the criterion func, but then mean on batch_size ?
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

        for ep in range(self.args.epoch):
            for iter, batch in enumerate(self.train_dataloader, start=1):
                if iter == self.args.n_iters:
                    break

                # split in train_iter? give directly batch to train_iter ?
                input_tensor = batch["ingr"].to(self.device)
                target_tensor = batch["target_instr"].to(self.device)
                target_length = batch["target_length"]  # .to(self.device)

                loss, decoded_words = self.train_iter(
                    input_tensor, target_tensor, target_length)
                print_loss_total += loss
                plot_loss_total += loss

                if iter % self.args.print_step == 0:
                    print_loss_avg = print_loss_total / self.args.print_step
                    print_loss_total = 0
                    print('Epoch {} {} ({} {}%) loss={}'.format(ep,timeSince(start, iter / self.args.n_iters),iter,int(iter / self.args.n_iters * 100),print_loss_avg))
                    print(" ".join(decoded_words[0]))

                    torch.save(self.state_dict(), os.path.join(
                            self.savepath, "model_{}_{}".format(datetime.now().strftime('%m-%d-%H-%M'),iter)))
                    if print_loss_avg<best_loss:
                        torch.save(self.state_dict(), os.path.join(
                            self.savepath, "best_model"))
                # if iter % plot_every == 0:
                #     plot_loss_avg = plot_loss_total / plot_every
                #     plot_losses.append(plot_loss_avg)
                #     plot_loss_total = 0
                #     showPlot(plot_losses)

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
            pair = random.choice(self.data.pairs)
            print('>', " ".join([ingr.name for ingr in pair[0]]))
            print('=', [" ".join(instr) for instr in pair[1]])
            loss, output_words, _ = self.evaluate(pair[0], pair[1])
            output_sentence = ' '.join(output_words[0])
            print('<', output_sentence)
            print('')

    def evalProcess(self, print_every=1000, plot_every=100):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        for iter, batch in enumerate(self.test_dataloader, start=1):
            # split in train_iter? give directly batch to train_iter ?
            input_tensor = batch["ingr"].to(self.device)
            target_tensor = batch["target_instr"].to(self.device)
            target_length = batch["target_length"]

            loss = self.train_iter(input_tensor, target_tensor, target_length)
            print_loss_total += loss
            plot_loss_total += loss

        print_loss_avg = print_loss_total / print_every
        print('Loss %.4f' % (print_loss_avg))


class Seq2seqAtt(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = AttnDecoderRNN(args, self.output_size)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)

    def evaluateAndShowAttention(self, input_sentence):
        loss, output_words, attentions = self.evaluate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)


class Seq2seqIngrAtt(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = IngrAttnDecoderRNN(args, self.output_size)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)


class Seq2seqIngrPairingAtt(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = PairAttnDecoderRNN(args, self.output_size, unk_token=self.train_dataset.UNK_token)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)

    def forward(self, input_tensor, target_tensor):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """
        self.batch_size = len(input_tensor)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        # encoder_outputs (hidden_size, batch) or (max_ingr,hidden) ??
        # encoder_hidden (1,hidden_size, batch)

        decoder_input = torch.tensor(
            [[self.train_dataset.SOS_token]*self.batch_size], device=self.device)
        decoder_hidden = encoder_hidden

        if self.training:
            use_teacher_forcing = True if random.random(
            ) < self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        decoded_words = [[] for i in range(self.batch_size)]
        decoder_attentions = torch.zeros(
            self.max_length, self.batch_size, self.decoder.pairAttention.pairings.top_k)
        decoder_outputs = torch.zeros(self.batch_size, self.max_length, len(
            self.train_dataset.vocab_tokens), device=self.device)

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, self.encoder.embedding, input_tensor)
                decoder_attentions[di] = decoder_attention.data
                decoder_outputs[:, di, :] = decoder_output
                decoder_input = target_tensor[:, di]

                topv, topi = decoder_output.topk(1)
                for batch_id, word_id in enumerate(topi):
                    decoded_words[batch_id].append(
                        self.train_dataset.vocab_tokens.idx2word[word_id.item()])

        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, self.encoder.embedding, input_tensor)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.topk(1)

                decoder_outputs[:, di, :] = decoder_output

                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[:, 0]
                if len(idx_end) == self.batch_size:
                    break

                for batch_id, word_id in enumerate(topi):
                    decoded_words[batch_id].append(
                        self.train_dataset.vocab_tokens.idx2word[word_id.item()])

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]
