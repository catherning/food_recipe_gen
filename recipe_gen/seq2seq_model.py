#!/usr/bin/env python
# coding: utf-8

# from __future__ import unicode_literals, print_function, division
from recipe_gen.seq2seq_utils import *
from recipe_1m_analysis.utils import MAX_LENGTH,MAX_INGR
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
    def __init__(self, input_size, hidden_size, max_ingr=MAX_INGR,device="cpu"):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.max_ingr = max_ingr

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_, hidden):
        embedded = self.embedding(input_).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def forward_all(self, input_tensor):
        # input_length = input_tensor.size(0)
        encoder_hidden = self.initHidden().to(self.device)
        encoder_outputs = torch.zeros(#self.batch_size, #XXX: ??
            self.max_ingr, self.hidden_size, device=self.device)

        for ei in range(self.max_ingr):
            # TODO: couldn't give directly all input_tensor, not step by step ???
            encoder_output, encoder_hidden = self.forward(
                input_tensor[:,ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        return encoder_outputs, encoder_hidden


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


        output = self.softmax(self.out(output[0])) # can use CrossEntropy instead if remove log softmax
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
    def __init__(self, input_size, hidden_size, output_size, batch_size, data, max_ingr=MAX_INGR,max_length=MAX_LENGTH, learning_rate=0.01, teacher_forcing_ratio=0.5, device="cpu"):
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

        self.encoder = EncoderRNN(input_size, hidden_size, max_ingr=max_ingr,device=device)
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
        # if target_tensor is not None:
        #     target_length = target_tensor.size(0)
        #     max_len = target_length
        # else:
        #     max_len = self.max_length
        
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
        decoder_outputs = torch.zeros(self.max_length,len(self.data.vocab_tokens), device=self.device)

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                decoder_outputs[di]=decoder_output
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_outputs[di]=decoder_output

                if topi.item() == self.data.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(
                        self.data.vocab_tokens.idx2word[topi.item()])

                decoder_input = topi.squeeze().detach()  # detach from history as input

        return decoder_outputs, decoded_words, None

    def train_iter(self, input_tensor, target_tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        decoded_outputs,decoded_words, _ = self.forward(input_tensor, target_tensor)
        loss = self.criterion(decoded_outputs,target_tensor)
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

        # TODO: use dataloader
        # training_pairs = [self.data.data[random.randint(
        #     0, len(self.data.data)-1)] for i in range(n_iters)]
        # for iter in range(1, n_iters + 1):
        #     training_pair = training_pairs[iter - 1]
        # TODO: first look at all dim size for encoder AND decoder, then convert with batch

        for i_batch, training_pair in enumerate(self.dataloader):
            if i_batch ==n_iters:
                break
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

    def evaluate(self, sentence, target=None):
        self.eval()
        with torch.no_grad():
            input_tensor = self.data.tensorFromSentence(
                self.data.vocab_ingrs, sentence).to(self.device)
            if target is not None:
                target_tensor = self.data.tensorFromSentence(self.data.vocab_tokens, target, instructions=True).to(self.device)
            else:
                target_tensor = None
            return self.forward(input_tensor, target_tensor)


    def evaluateRandomly(self, n=10):
        for i in range(n):
            pair = random.choice(self.data.pairs)
            print('>', " ".join(pair[0]))
            print('=', [" ".join(instr) for instr in pair[1]])
            loss, output_words, _ = self.evaluate(pair[0], pair[1])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')


class Seq2seqAtt(Seq2seq):
    def __init__(self, input_size, hidden_size, output_size, batch_size, data, max_length=MAX_LENGTH, learning_rate=0.01, teacher_forcing_ratio=0.5, device="cpu"):
        super().__init__(input_size, hidden_size, output_size, batch_size, data, max_length=max_length,
                         learning_rate=learning_rate, teacher_forcing_ratio=teacher_forcing_ratio, device=device)

        self.decoder = AttnDecoderRNN(
            hidden_size, output_size, dropout_p=0.1, max_length=max_length)
        self.decoder_optimizer = optim.SGD(
            self.decoder.parameters(), lr=learning_rate)

    def forward(self, input_tensor, target_tensor):
        if target_tensor is not None:
            target_length = target_tensor.size(0)
            max_len = target_length
        else:
            max_len = self.max_length

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
        decoder_attentions = torch.zeros(self.max_length, self.max_length)
        decoder_outputs = torch.zeros(max_len,len(self.data.vocab_tokens), device=self.device)

        if use_teacher_forcing:
            for di in range(max_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                decoder_outputs[di]=decoder_output
                decoder_input = target_tensor[di]

        else:
            for di in range(max_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.topk(1)

                decoder_outputs[di]=decoder_output

                if topi.item() == self.data.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(
                        self.data.vocab_tokens.idx2word[topi.item()])

                decoder_input = topi.squeeze().detach()  # detach from history as input

        return decoder_outputs,decoded_words, decoder_attentions[:di + 1]

    def evaluateAndShowAttention(self, input_sentence):
        loss, output_words, attentions = self.evaluate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)
