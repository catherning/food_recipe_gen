#!/usr/bin/env python
# coding: utf-8

# from __future__ import unicode_literals, print_function, division
from recipe_gen.seq2seq_utils import *
from io import open
import unicodedata
import re
import random
from functools import reduce
import os
import pickle
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import sys
sys.path.insert(0, "D:\\Documents\\THU\\food_recipe_gen")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_, hidden):
        embedded = self.embedding(input_).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(DecoderRNN):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__(hidden_size, output_size)
        self.attention = Attention(
            hidden_size, dropout_p=dropout_p, max_length=max_length)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)

        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


class Attention(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, embedded, hidden, encoder_outputs):
        """key:encoder_outputs
        value:hidden
        query: embedded
        """
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        return output, attn_weights


class Seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, data, max_length=MAX_LENGTH, learning_rate=0.01, teacher_forcing_ratio=0.5, attention=True,device="cpu"):
        super().__init__()
        # all params in arg storage object ?
        self.max_length = max_length
        self.data = data
        self.device = device

        self.encoder = EncoderRNN(input_size, hidden_size)
        self.attention = attention
        if attention:
            self.decoder = AttnDecoderRNN(
                hidden_size, output_size, dropout_p=0.1, max_length=max_length)
        else:
            self.decoder = DecoderRNN(hidden_size, output_size)

        self.encoder_optimizer = optim.SGD(
            self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(
            self.decoder.parameters(), lr=learning_rate)

        # Training param
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.learning_rate = learning_rate
        self.criterion = nn.NLLLoss()

    def forward(self, input_tensor, target_tensor):
        encoder_hidden = self.encoder.initHidden().to(self.device)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.data.SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random(
        ) < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            if self.attention:
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += self.criterion(decoder_output, target_tensor[di])
                    decoder_input = target_tensor[di]  # Teacher forcing
            else:
                for di in range(target_length):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += self.criterion(decoder_output, target_tensor[di])
                    decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            if self.attention:
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss += self.criterion(decoder_output, target_tensor[di])
                    if decoder_input.item() == self.data.EOS_token:
                        break
            else:
                for di in range(target_length):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss += self.criterion(decoder_output, target_tensor[di])
                    if decoder_input.item() == self.data.EOS_token:
                        break
        return loss/target_length

    def train_iter(self, input_tensor, target_tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        loss = self.forward(input_tensor, target_tensor)

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def train_process(self, n_iters, print_every=1000, plot_every=100):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        # TODO: use dataloader
        training_pairs = [self.data.data[random.randint(
            0, len(self.data.data)-1)] for i in range(n_iters)]

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0].to(self.device)
            target_tensor = training_pair[1].to(self.device)

            loss = self.train_iter(input_tensor, target_tensor)
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

    def evaluate(self, sentence):
        # TODO use forward
        with torch.no_grad():
            input_tensor = self.data.tensorFromSentence(
                self.data.vocab_ingrs, sentence).to(self.device)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden().to(self.device)

            encoder_outputs = torch.zeros(
                self.max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                              encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor(
                [[self.data.SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.data.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(
                        self.data.vocab_tokens.idx2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def evaluateRandomly(self, n=10):
        for i in range(n):
            pair = random.choice(self.data.pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    def evaluateAndShowAttention(self, input_sentence):
        output_words, attentions = self.evaluate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)
