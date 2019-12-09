#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import re
import sys
import time
import unicodedata
from functools import reduce
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from recipe_1m_analysis.utils import MAX_INGR, MAX_LENGTH
from recipe_gen.network import *
from recipe_gen.seq2seq_utils import *

sys.path.insert(0, "D:\\Documents\\THU\\food_recipe_gen")


class Seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, data, max_ingr=MAX_INGR, max_length=MAX_LENGTH, learning_rate=0.01, teacher_forcing_ratio=0.5, device="cpu"):
        super().__init__()
        # all params in arg storage object ?
        self.max_length = max_length
        self.max_ingr = max_ingr
        self.data = data
        self.dataloader = torch.utils.data.DataLoader(data,
                                                      batch_size=batch_size, shuffle=True,
                                                      num_workers=4)
        self.batch_size = batch_size
        self.device = device

        self.encoder = EncoderRNN(
            input_size, hidden_size, batch_size, max_ingr=max_ingr, device=device)
        self.decoder = DecoderRNN(hidden_size, output_size, batch_size)

        self.encoder_optimizer = optim.SGD(
            self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(
            self.decoder.parameters(), lr=learning_rate)

        # Training param
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.learning_rate = learning_rate
        self.criterion = nn.NLLLoss()

    def forward(self, input_tensor, target_tensor):

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        decoder_input = torch.tensor(
            [[self.data.SOS_token]], device=self.device)
        decoder_hidden = encoder_hidden

        if self.training:
            use_teacher_forcing = True if random.random(
            ) < self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        decoded_words = []
        decoder_outputs = torch.zeros(
            len(self.data.vocab_tokens), self.max_length, device=self.device)

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                decoder_outputs[:, di] = decoder_output
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_outputs[:, di] = decoder_output

                if topi.item() == self.data.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(
                        self.data.vocab_tokens.idx2word[topi.item()])

                decoder_input = topi.squeeze().detach()  # detach from history as input

        return decoder_outputs, decoded_words, None

    def train_iter(self, input_tensor, target_tensor, target_length):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        decoded_outputs, decoded_words, _ = self.forward(
            input_tensor, target_tensor)
        aligned_outputs = flattenSequence(decoded_outputs, target_length)
        aligned_target = flattenSequence(target_tensor[:, 1:], target_length)
        loss = self.criterion(aligned_outputs, aligned_target)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def train_process(self, n_iters, print_every=1000, plot_every=100):
        self.train()
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        for iter, batch in enumerate(self.dataloader, start=1):
            if iter == n_iters:
                break

            # split in train_iter? give directly batch to train_iter ?
            input_tensor = batch["ingr"].to(self.device)
            target_tensor = batch["target_instr"].to(self.device)
            target_length = batch["target_length"]  # .to(self.device)

            loss = self.train_iter(input_tensor, target_tensor, target_length)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)

    def evaluate(self, sentence, target=None):
        self.eval()
        with torch.no_grad():
            input_tensor = self.data.tensorFromSentence(
                self.data.vocab_ingrs, sentence).to(self.device).view(1, -1)
            if target is not None:
                target_tensor = self.data.tensorFromSentence(
                    self.data.vocab_tokens, target, instructions=True).to(self.device).view(1, -1)
            else:
                target_tensor = None
            return self.forward(input_tensor, target_tensor)

    def evaluateRandomly(self, n=10):
        for i in range(n):
            pair = random.choice(self.data.pairs)
            print('>', " ".join(pair[0]))
            print('=', [" ".join(instr) for instr in pair[1]])
            loss, output_words, _ = self.evaluate(pair[0], pair[1])
            output_sentence = ' '.join(output_words[0])
            print('<', output_sentence)
            print('')

    def evalProcess(self):
        pass


class Seq2seqAtt(Seq2seq):
    def __init__(self, input_size, hidden_size, output_size, batch_size, data, max_ingr=MAX_INGR, max_length=MAX_LENGTH, learning_rate=0.01, teacher_forcing_ratio=0.5, device="cpu"):
        super().__init__(input_size, hidden_size, output_size, batch_size, data, max_ingr=max_ingr, max_length=max_length,
                         learning_rate=learning_rate, teacher_forcing_ratio=teacher_forcing_ratio, device=device)

        self.decoder = AttnDecoderRNN(
            hidden_size, output_size, batch_size, dropout_p=0.1, max_ingr=max_ingr, max_length=max_length)
        self.decoder_optimizer = optim.SGD(
            self.decoder.parameters(), lr=learning_rate)

    def forward(self, input_tensor, target_tensor):

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        decoder_input = torch.tensor(
            [[self.data.SOS_token]*self.batch_size], device=self.device)
        decoder_hidden = encoder_hidden

        if self.training:
            use_teacher_forcing = True if random.random(
            ) < self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        decoded_words = [[] for i in range(self.batch_size)]
        decoder_attentions = torch.zeros(
            self.max_length, self.batch_size, self.max_ingr)
        decoder_outputs = torch.zeros(self.batch_size, self.max_length, len(
            self.data.vocab_tokens), device=self.device)

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                decoder_outputs[:, di, :] = decoder_output
                decoder_input = target_tensor[:, di]

        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.topk(1)

                decoder_outputs[:, di, :] = decoder_output

                idx_end = (topi == self.data.EOS_token).nonzero()[:, 0]
                if len(idx_end) == self.batch_size:
                    break

                for batch_id, word_id in enumerate(topi):
                    decoded_words[batch_id].append(
                        self.data.vocab_tokens.idx2word[word_id.item()])

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]

    def evaluateAndShowAttention(self, input_sentence):
        loss, output_words, attentions = self.evaluate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)
