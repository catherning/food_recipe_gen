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
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from recipe_1m_analysis.utils import MAX_INGR, MAX_LENGTH
from recipe_gen.network import *
from recipe_gen.seq2seq_utils import *

sys.path.insert(0, "D:\\Documents\\THU\\food_recipe_gen")


class Seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, data, max_ingr=MAX_INGR, max_length=MAX_LENGTH, learning_rate=0.01, teacher_forcing_ratio=0.5, device="cpu",savepath="./results/"):
        super().__init__()
        # all params in arg storage object ?
        self.max_length = max_length
        self.max_ingr = max_ingr
        self.data = data

        train_size = int(0.8 * len(self.data))
        test_size = len(self.data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.data, [train_size, test_size])

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=batch_size, shuffle=True,
                                                      num_workers=4)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=batch_size, shuffle=True,
                                                      num_workers=4)

        self.batch_size = batch_size
        self.device = device
        self.savepath = savepath

        self.encoder = EncoderRNN(
            input_size, hidden_size, batch_size, max_ingr=max_ingr, device=device)
        self.decoder = DecoderRNN(hidden_size, output_size, batch_size)

        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=learning_rate)

        # Training param
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.learning_rate = learning_rate
        self.criterion = nn.NLLLoss()

    def forward(self, input_tensor, target_tensor):
        self.batch_size = len(input_tensor)
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
        decoder_outputs = torch.zeros(self.batch_size, self.max_length, len(
            self.data.vocab_tokens), device=self.device)

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                decoder_outputs[:, di] = decoder_output
                decoder_input = target_tensor[:, di].view(1,-1)   # Teacher forcing

                topv, topi = decoder_output.topk(1)
                for batch_id, word_id in enumerate(topi):
                    decoded_words[batch_id].append(
                        self.data.vocab_tokens.idx2word[word_id.item()])

        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_outputs[:, di, :] = decoder_output

                idx_end = (topi == self.data.EOS_token).nonzero()[:, 0]
                if len(idx_end) == self.batch_size:
                    break

                for batch_id, word_id in enumerate(topi):
                    decoded_words[batch_id].append(
                        self.data.vocab_tokens.idx2word[word_id.item()])
        
                decoder_input = topi.squeeze().detach().view(1,-1)  # detach from history as input

        return decoder_outputs, decoded_words, None

    def train_iter(self, input_tensor, target_tensor, target_length):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        decoded_outputs, decoded_words, _ = self.forward(
            input_tensor, target_tensor)
        aligned_outputs = flattenSequence(decoded_outputs, target_length)
        aligned_target = flattenSequence(target_tensor, target_length)
        loss = self.criterion(aligned_outputs, aligned_target)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item(),decoded_words

    def train_process(self, n_iters, print_every=1000, plot_every=100):
        self.train()
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        for iter, batch in enumerate(self.train_dataloader, start=1):
            if iter == n_iters:
                break

            # split in train_iter? give directly batch to train_iter ?
            input_tensor = batch["ingr"].to(self.device)
            target_tensor = batch["target_instr"].to(self.device)
            target_length = batch["target_length"]  # .to(self.device)

            loss,decoded_words = self.train_iter(input_tensor, target_tensor, target_length)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print(f"{timeSince(start, iter / n_iters)} ({iter} {int(iter / n_iters * 100)}%) loss={print_loss_avg})")
                print(decoded_words[0])
                torch.save(self.state_dict(), os.path.join(self.savepath,f"model_{datetime.now().strftime('%m-%d-%H-%M')}_{iter}"))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)

    def evaluate(self, sentence, target=None):
        self.eval()
        with torch.no_grad():
            input_tensor,_ = self.data.tensorFromSentence(
                self.data.vocab_ingrs, sentence)
            input_tensor = input_tensor.view(1, -1).to(self.device)
            if target is not None:
                target_tensor,_ = self.data.tensorFromSentence(
                    self.data.vocab_tokens, target, instructions=True)
                target_tensor=target_tensor.view(1, -1).to(self.device)
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


    def evalProcess(self, print_every=1000, plot_every=100):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        for iter, batch in enumerate(self.test_dataloader, start=1):

            # split in train_iter? give directly batch to train_iter ?
            input_tensor = batch["ingr"].to(self.device)
            target_tensor = batch["target_instr"].to(self.device)
            target_length = batch["target_length"]  # .to(self.device)

            loss = self.train_iter(input_tensor, target_tensor, target_length)
            print_loss_total += loss
            plot_loss_total += loss

        print_loss_avg = print_loss_total / print_every
        print('Loss %.4f' % (print_loss_avg))


class Seq2seqAtt(Seq2seq):
    def __init__(self, input_size, hidden_size, output_size, batch_size, data, max_ingr=MAX_INGR, max_length=MAX_LENGTH, learning_rate=0.01, teacher_forcing_ratio=0.5, device="cpu",savepath="./results/"):
        super().__init__(input_size, hidden_size, output_size, batch_size, data, max_ingr=max_ingr, max_length=max_length,
                         learning_rate=learning_rate, teacher_forcing_ratio=teacher_forcing_ratio, device=device,savepath=savepath)

        self.decoder = AttnDecoderRNN(
            hidden_size, output_size, batch_size, dropout_p=0.1, max_ingr=max_ingr, max_length=max_length)
        self.decoder_optimizer = optim.SGD(
            self.decoder.parameters(), lr=learning_rate)

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
            [[self.data.SOS_token]*self.batch_size], device=self.device)
        decoder_hidden = encoder_hidden
        # decoder_input final (<max_len,batch)

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
                decoder_input = target_tensor[:, di].view(1,-1) # for di=0, takes <sos> again!!! 
                # => remove sos in tensorFromSentence in utils
                
                topv, topi = decoder_output.topk(1)
                for batch_id, word_id in enumerate(topi):
                    decoded_words[batch_id].append(
                        self.data.vocab_tokens.idx2word[word_id.item()])

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

                decoder_input = topi.squeeze().detach().view(1, -1)  # detach from history as input

        return decoder_outputs, decoded_words, decoder_attentions[:di + 1]

    def evaluateAndShowAttention(self, input_sentence):
        loss, output_words, attentions = self.evaluate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)


class Seq2seqIngrAtt(Seq2seq):
    def __init__(self, input_size, hidden_size, output_size, batch_size, data,pairing_path, max_ingr=MAX_INGR, max_length=MAX_LENGTH, learning_rate=0.01, teacher_forcing_ratio=0.5, device="cpu",savepath="./results/"):
        super().__init__(input_size, hidden_size, output_size, batch_size, data, max_ingr=max_ingr, max_length=max_length,
                         learning_rate=learning_rate, teacher_forcing_ratio=teacher_forcing_ratio, device=device,savepath=savepath)

        self.decoder = PairAttnDecoderRNN(pairing_path,
            hidden_size, output_size, batch_size, dropout_p=0.1, max_ingr=max_ingr, max_length=max_length)
        self.decoder_optimizer = optim.SGD(
            self.decoder.parameters(), lr=learning_rate)

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
                    decoder_input, decoder_hidden, encoder_outputs,self.encoder.embedding)
                decoder_attentions[di] = decoder_attention.data
                decoder_outputs[:, di, :] = decoder_output
                decoder_input = target_tensor[:, di]

        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs,self.encoder.embedding)
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