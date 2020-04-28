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
from operator import itemgetter
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

sys.path.insert(0, os.getcwd())

from recipe_gen.seq2seq_utils import *
from recipe_gen.network import *
from KitcheNette_master.unk_pairs_gen import getMainIngr

class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_length = args.max_length
        self.max_ingr = args.max_ingr
        self.train_dataset = RecipesDataset(args)
        self.test_dataset = RecipesDataset(args, train=False)
        self.input_size = input_size = len(self.train_dataset.vocab_ingrs)
        self.output_size = output_size = len(self.train_dataset.vocab_tokens)
        self.hidden_size = args.hidden_size
        self.hierarchical = True if "Hierarchical" in self.args.model_name else False
   

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=args.batch_size, shuffle=True,
                                                            num_workers=0)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=args.batch_size, shuffle=False,
                                                           num_workers=0)
        self.test_dataset.id2index = list(map(itemgetter('id'), self.test_dataset.data))

        self.batch_size = args.batch_size
        self.device = args.device
        self.savepath = args.saving_path
        self.logger = args.logger
        self.train_mode = args.train_mode

        self.encoder = EncoderRNN(args, input_size, args.ingr_embed)
        self.decoder = DecoderRNN(args, output_size)

        # TODO: change optim: self.optimizers  = optim.Adam(self.parameters())
        # or Adam((self.encoder.param,self.decoder.param) directly ?
        # => no need for optim_list ?
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=args.learning_rate)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list = [self.encoder_optimizer, self.decoder_optimizer]
        
        self.encoder_fusion = nn.Sequential(nn.Linear((2 if self.args.bidirectional else 1) * args.hidden_size, args.hidden_size),
                                            nn.Tanh())
        self.fusion_optim = optim.Adam(
            (self.encoder_fusion.parameters()), lr=args.learning_rate)
        self.optim_list.append(self.fusion_optim)

        # Training param
        self.decay_factor = args.decay_factor
        self.learning_rate = args.learning_rate
        self.criterion = nn.CrossEntropyLoss(reduction="sum",ignore_index=self.train_dataset.PAD_token)
        
        self.training_losses =[]

        self.paramLogging()
        
    def paramLogging(self):
        for k, v in self.args.defaults.items():
            try:
                if v is None or getattr(self.args, k) != v:
                    self.logger.info("{} = {}".format(
                        k, getattr(self.args, k)))
            except AttributeError:
                continue
        
    def train_iter(self, batch, iter):
        # target_length = batch["target_length"]
        target_tensor = batch["target_instr"].to(self.device)
        self.batch_size = batch_size = target_tensor.shape[0]

        decoder_outputs, decoded_words, _ = self.forward(batch, iter=iter)
        # TODO: change flattenSeq if hierarchical
        # aligned_outputs = flattenSequence(decoder_outputs, target_length)
        # aligned_target = flattenSequence(target_tensor, target_length)
        
        if self.hierarchical:
            aligned_outputs = decoder_outputs.view(
                batch_size*self.max_length*self.max_step, self.output_size)
            aligned_target = target_tensor.view(batch_size*self.max_length*self.max_step)
        else:
            aligned_outputs = decoder_outputs.view(
                batch_size*self.max_length, self.output_size)
            aligned_target = target_tensor.view(batch_size*self.max_length)
        
        loss = self.criterion(
            aligned_outputs, aligned_target)/batch_size

        return loss, decoded_words

    def train_process(self):
        start = time.time()
        plot_losses = self.training_losses
        print_loss_total = 0
        plot_loss_total = 0
        best_loss = math.inf

        def lmbda(epoch): return 0.95
        scheduler_list = [torch.optim.lr_scheduler.MultiplicativeLR(
            optim, lr_lambda=lmbda) for optim in self.optim_list]
        plot_losses = []
        for ep in range(self.args.begin_epoch, self.args.epoch+1):
            self.train()
            for iter, batch in enumerate(self.train_dataloader, start=1):
                if iter == self.args.n_iters:
                    break

                for optim in self.optim_list:
                    optim.zero_grad()

                loss, decoded_words = self.train_iter(batch, iter)

                if iter % self.args.update_step:
                    loss.backward()

                    for optim in self.optim_list:
                        optim.step()

                print_loss_total += loss.detach()
                plot_loss_total += loss.detach()

                if iter % max(self.args.print_step, self.args.n_iters//10) == 0:
                    print_loss_avg = print_loss_total / \
                        max(self.args.print_step, self.args.n_iters//10)
                    plot_losses.append(print_loss_avg)
                    print_loss_total = 0
                    self.logger.info('Epoch {} {} ({} {}%) loss={}'.format(ep, timeSince(
                        start, iter / self.args.n_iters), iter, int(iter / self.args.n_iters * 100), print_loss_avg))

                    try:
                        self.logger.info(
                            "Generated =  "+" ".join(decoded_words[0][:batch["target_length"][0]]))
                        self.logger.info("Target =  " + " ".join([self.train_dataset.vocab_tokens.idx2word[word.item(
                        )] for word in batch["target_instr"][0,:batch["target_length"][0]]]))
                    except TypeError:
                        self.logger.info("Generated =  {}".format([" ".join(sent[:batch["target_length"][0,i]]) 
                                                                   for i,sent in enumerate(decoded_words[0])]))
                        self.logger.info("Target = {}".format([" ".join([self.train_dataset.vocab_tokens.idx2word[word.item()] 
                                                                         for word in sent[:batch["target_length"][0,i]]]) 
                                                                         for i,sent in enumerate(batch["target_instr"][0])]))


            torch.save({
                'epoch': ep,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': [optim.state_dict() for optim in self.optim_list],
                'loss': loss,
                'loss_list': plot_losses
            }, os.path.join(self.savepath, "train_model_{}_{}.tar".format(datetime.now().strftime('%m-%d-%H-%M'), ep)))

            val_loss = self.evalProcess()
            if val_loss < best_loss:
                self.logger.info("Best model so far, saving it.")
                torch.save(self.state_dict(), os.path.join(
                    self.savepath, "best_model"))
                best_loss = val_loss

            for scheduler in scheduler_list:
                scheduler.step()
                
        showPlot(plot_losses,self.savepath)
        
    def evalProcess(self):
        """
        Eval method called during training after each epoch
        """
        self.eval()
        start = time.time()
        print_loss_total = 0
        with torch.no_grad():
            for iter, batch in enumerate(self.test_dataloader, start=1):
                loss, _ = self.train_iter(batch, iter)
                print_loss_total += loss.detach()

                if iter % max(self.args.print_step, len(self.test_dataloader)//(10*self.test_dataloader.batch_size)) == 0:
                    print("Eval Current loss = {}".format(
                        print_loss_total/iter))

            print_loss_avg = print_loss_total / iter
            self.logger.info("Eval loss = {}".format(print_loss_avg))
        return print_loss_avg
    
    def evalOutput(self):
        """
        Evaluate on dev dataset and prints generated to console and into logfile
        """
        self.eval()
        start = time.time()
        print_loss_total = 0

        with torch.no_grad():
            for iter, batch in enumerate(self.test_dataloader, start=1):
                loss, output_words = self.train_iter(batch, iter)

                for i, ex in enumerate(output_words):
                    try:
                        self.logger.info(batch["id"][i]+" "+' '.join(ex))
                    except TypeError:
                        self.logger.info(
                            batch["id"][i]+" "+"|".join([' '.join(sent) for sent in ex]))

                print_loss_total += loss.detach()

                if iter % self.args.print_step == 0:
                    print("Current loss = {}".format(print_loss_total/iter))

            print_loss_avg = print_loss_total / iter
            self.logger.info("Eval loss = {}".format(print_loss_avg))

    def evaluateFromText(self, sample):
        self.eval()
        with torch.no_grad():
            input_tensor, _ = self.train_dataset.tensorFromSentence(
                self.train_dataset.vocab_ingrs, sample["ingr"])
            input_tensor = input_tensor.view(1, -1).to(self.device)

            try:
                target_tensor, _ = self.train_dataset.tensorFromSentence(
                    self.train_dataset.vocab_tokens, sample["target"], instructions=True)
                target_tensor = target_tensor.view(1, -1).to(self.device)
            except KeyError:
                target_tensor = None

            try:
                title_tensor, _ = self.train_dataset.tensorFromSentence(
                    self.train_dataset.vocab_tokens, sample["title"])
                title_tensor = title_tensor.view(1, -1).to(self.device)
            except KeyError:
                title_tensor = None
            
            try:
                cuis_tensor = torch.LongTensor([self.train_dataset.vocab_cuisine.word2idx[sample["cuisine"]]])
            except (KeyError,AttributeError):
                cuis_tensor = None

            batch = {"ingr": input_tensor,
                     "target_instr": target_tensor, 
                     "title": title_tensor,
                     "cuisine":cuis_tensor}
            return self.forward(batch)

    def evalSample(self,sample):
        for k in ["ingr","target_instr"]+["title"]*("title" in sample)+["cuisine"]*("cuisine" in sample):
            sample[k] = sample[k].unsqueeze(0)
        _, output_words, att_data = self.forward(sample)

        try:
            output_sentence = ' '.join(output_words[0])
        except TypeError:
            output_sentence = "|".join(
                [' '.join(sent) for sent in output_words[0]])

        or_sent = " ".join([self.train_dataset.vocab_ingrs.idx2word[ingr.item()][0] for ingr in sample["ingr"][0]])
        self.logger.info(
            "Input: "+ or_sent)
        self.logger.info(
            "Target: "+str([" ".join([self.train_dataset.vocab_tokens.idx2word[word.item()] for word in instr if word.item() != 0]) for instr in sample["target_instr"]]))
        self.logger.info("Generated: "+output_sentence)
        
        try:
            attentions = att_data["attentions"][:,0]
            comp_ingr_id = att_data["comp_ingrs"][:,0]
            focused_ingrs_id = att_data["focused_ingr"][:,0]
            focused_ingrs = [self.vocab_main_ingr.idx2word.get(ingr.item(),'<unk>')[0] for ingr in focused_ingrs_id]
            comp_ingr = [' '.join([self.vocab_main_ingr.idx2word.get(ingr.item(),'<unk>')[0] for ingr in comp_ingr_id[i]]) for i in range(comp_ingr_id.shape[0])]
            showPairingAttention(comp_ingr, focused_ingrs, output_words[0], attentions,self.savepath,name=sample["id"][:3])
        except AttributeError:
            showSentAttention(or_sent, output_words[0], attentions,self.savepath,name=sample["id"][:3])
        except TypeError:
            pass
            
    def evaluateRandomly(self, n=10):
        self.eval()
        self.batch_size = 1
        with torch.no_grad():
            for i in range(n):
                sample = random.choice(self.test_dataset.data)
                self.evalSample(sample)
    
    def evalFromId(self,id):
        self.eval()
        self.batch_size = 1
        with torch.no_grad():
            idx = self.test_dataset.id2index.index(id)
            sample = self.test_dataset.data[idx]            
            self.evalSample(sample)
                            
class Seq2seq(BaseModel):
    def __init__(self, args):
        super().__init__(args)

    def addAttention(self, di, decoder_attentions, cur_attention):
        if cur_attention is not None:
            decoder_attentions[di] = cur_attention.data
        return decoder_attentions

    def getSamplingProba(self, iter):
        if self.args.scheduled_sampling and self.training:
            sampling_proba = 1-inverse_sigmoid_decay(
                self.decay_factor, iter)
        elif not self.args.scheduled_sampling and self.training:
            sampling_proba = 0
        else:
            sampling_proba = 1
        return sampling_proba


    def initForward(self, input_tensor, pairing=False):
        # XXX: should be able not to reassign if do view with correct hidden size instead
        batch_size = self.batch_size
        decoder_input = torch.tensor(
            [[self.train_dataset.SOS_token]*batch_size], device=self.device)
        # decoder_input final (<max_len,batch)

        decoded_words = [[] for i in range(batch_size)]
        decoder_outputs = torch.zeros(batch_size, self.max_length, len(
            self.train_dataset.vocab_tokens), device=self.device)

        if pairing:
            decoder_attentions = torch.zeros(
                self.max_length, batch_size, self.decoder.pairAttention.pairings.top_k)
            comp_ingrs = torch.zeros(self.max_length,batch_size,self.decoder.pairAttention.pairings.top_k,dtype=torch.int)
            focused_ingrs = torch.zeros(self.max_length,batch_size,dtype=torch.int)
            return decoder_input, decoded_words, decoder_outputs, decoder_attentions,comp_ingrs,focused_ingrs
        else:
            decoder_attentions = torch.zeros(
                self.max_length, batch_size, self.max_ingr)
            return decoder_input, decoded_words, decoder_outputs, decoder_attentions


    def forwardDecoderStep(self, decoder_input, decoder_hidden,
                           encoder_outputs, di, decoder_attentions, decoder_outputs, decoded_words):
        decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs)  # can remove encoder_outputs ? not used in decoder
        decoder_outputs[:, di] = decoder_output
        decoder_attentions = self.addAttention(
            di, decoder_attentions, decoder_attention)
        topi = samplek(self,decoder_output, decoded_words,self.train_dataset.vocab_tokens.idx2word)
        return decoder_attentions, decoder_hidden, topi

    def forward(self, batch, iter=iter):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        return:
        decoder_outputs: (batch, max_len, size voc)
        decoder_words: final (batch, max_len)
        decoder_attentions: (<max_len, batch, max_ingr)
        """

        input_tensor = batch["ingr"].to(self.device)
        try:
            target_tensor = batch["target_instr"].to(
                self.device)  # (batch,max_step,max_length) ?
        except AttributeError:
            target_tensor = None

        decoder_input, decoded_words, decoder_outputs, decoder_attentions = self.initForward(
            input_tensor)

        # encoder_outputs (max_ingr,batch, 2*hidden_size)
        # encoder_hidden (1, batch, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        encoder_hidden = self.encoder_fusion(encoder_hidden)
        decoder_hidden = encoder_hidden  # (num_layers, batch, hidden_size)

        sampling_proba = self.getSamplingProba(iter)

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

        return decoder_outputs, decoded_words, None


class Seq2seqAtt(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoder = AttnDecoderRNN(args, self.output_size)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=args.learning_rate)
        self.optim_list[1] = self.decoder_optimizer

    def evaluateAndShowAttention(self, sample):
        _, output_words, attentions, comp_ingr_id = self.evaluateFromText(sample)
        self.logger.info('input = ' + ' '.join(sample["ingr"]))
        self.logger.info('output = ' + ' '.join(output_words[0]))
        # showAttention(sample["ingr"], output_words, attentions[:,0],self.savepath)
        try:
            comp_ingr_id = comp_ingr_id[:,0]
            attentions = attentions[:,0]
            comp_ingr = [' '.join([self.vocab_main_ingr.idx2word.get(ingr.item(),'<unk>')[0] for ingr in comp_ingr_id[i]]) for i in range(comp_ingr_id.shape[0])]
            showPairingAttention(comp_ingr, output_words[0], attentions,self.savepath,name="user_input")
        except AttributeError:
            showSentAttention(sample["ingr"], output_words[0], attentions,self.savepath,name="user_input")


class Seq2seqTrans(Seq2seq):
    def __init__(self, args):
        super().__init__(args)

        self.decoderLayer = nn.TransformerDecoderLayer(
            d_model=args.hidden_size, nhead=8)
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
        
        self.vocab_main_ingr = getMainIngr(self.train_dataset.vocab_ingrs)
        # for i in range(4):
        #     self.vocab_main_ingr.add_word(self.train_dataset.vocab_ingrs.idx2word[i],i)

    def forwardDecoderStep(self, decoder_input, decoder_hidden,
                           encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words,comp_ingrs,focused_ingrs):
        decoder_output, decoder_hidden, decoder_attention, comp_ingr,focused_ingr = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs, self.encoder.embedding, input_tensor)

        decoder_attentions = self.addAttention(
            di, decoder_attentions, decoder_attention)
        decoder_outputs[:, di] = decoder_output
        topi = samplek(self,decoder_output, decoded_words,self.train_dataset.vocab_tokens.idx2word)
        comp_ingrs[di]=comp_ingr
        focused_ingrs[di]= focused_ingr
        return decoder_attentions, decoder_hidden, topi, comp_ingrs,focused_ingrs

    def forward(self, batch, iter=iter):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        encoder_outputs (max_ingr, batch, hidden_size*2) 
        encoder_hidden (num_layers, batch, hidden_size)

        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """

        input_tensor = batch["ingr"].to(self.device)
        try:
            target_tensor = batch["target_instr"].to(self.device)
        except AttributeError:
            Warning("Evaluation mode: only taking ingredient list as input")

        # Encoder
        decoder_input, decoded_words, decoder_outputs, decoder_attentions,comp_ingrs,focused_ingrs = self.initForward(
            input_tensor, pairing=True)

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)

        decoder_hidden = self.encoder_fusion(encoder_hidden)

        # Scheduled sampling
        sampling_proba = self.getSamplingProba(iter)
        
        # Decoder part
        for di in range(self.max_length):
            decoder_attentions, decoder_hidden, topi,comp_ingrs,focused_ingrs = self.forwardDecoderStep(decoder_input, decoder_hidden,
                                                                               encoder_outputs, input_tensor, di, decoder_attentions, 
                                                                               decoder_outputs, decoded_words,comp_ingrs,focused_ingrs)
            if random.random() < sampling_proba:
                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[:, 0]
                if len(idx_end) == self.batch_size:
                    break

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input
            else:
                decoder_input = target_tensor[:, di].view(1, -1)

        return decoder_outputs, decoded_words, {"attentions": decoder_attentions[:di + 1],
                                                "comp_ingrs":comp_ingrs[:di+1],
                                                "focused_ingr":focused_ingrs[:di+1]}


class Seq2seqTitlePairing(Seq2seqIngrPairingAtt):
    def __init__(self, args):
        super().__init__(args)
        self.title_encoder = EncoderRNN(args, self.output_size, args.title_embed)
        # output because tok of  title are in vocab_toks
        self.title_optimizer = optim.Adam(
            self.title_encoder.parameters(), lr=args.learning_rate)
        self.optim_list.append(self.title_optimizer)

        self.encoder_fusion = nn.Sequential(nn.Linear((2 if self.args.bidirectional else 1)* 2 * args.hidden_size, args.hidden_size),
                                            nn.Tanh())
        self.fusion_optim = optim.Adam(
            (self.encoder_fusion.parameters()), lr=args.learning_rate)
        self.optim_list.append(self.fusion_optim)

    def forward(self, batch, iter=iter):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)
        
        encoder_outputs (max_ingr,hidden_size, batch)
        encoder_hidden (1,hidden_size, batch)

        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """
        input_tensor = batch["ingr"].to(self.device)
        decoder_input, decoded_words, decoder_outputs, decoder_attentions,comp_ingrs,focused_ingrs = self.initForward(
            input_tensor, pairing=True)

        try:
            target_tensor = batch["target_instr"].to(self.device)
        except AttributeError:
            Warning("Evaluation mode: only taking ingredient list as input")

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        
        title_tensor = batch["title"].to(self.device)
        title_encoder_outputs, title_encoder_hidden = self.title_encoder.forward_all(
            title_tensor)

        decoder_hidden = torch.cat(
            (encoder_hidden, title_encoder_hidden), dim=2)
        decoder_hidden = self.encoder_fusion(decoder_hidden)

        sampling_proba = self.getSamplingProba(iter)

        for di in range(self.max_length):
            decoder_attentions, decoder_hidden, topi,comp_ingr = self.forwardDecoderStep(
                decoder_input, decoder_hidden, encoder_outputs, input_tensor, di, decoder_attentions, decoder_outputs, decoded_words)
            
            if random.random() < sampling_proba:
                idx_end = (topi == self.train_dataset.EOS_token).nonzero()[:, 0]
                if len(idx_end) == self.batch_size:
                    break

                decoder_input = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input
            else:
                decoder_input = target_tensor[:, di].view(1, -1)

        return decoder_outputs, decoded_words, {"attentions": decoder_attentions[:di + 1],
                                                "comp_ingrs":comp_ingrs[:di+1],
                                                "focused_ingr":focused_ingrs[:di+1]}


class Seq2seqCuisinePairing(Seq2seqIngrPairingAtt):
    def __init__(self, args):
        super().__init__(args)
        self.hidden_size = args.hidden_size
        self.cuis_embed = args.cuisine_embed

        # self.cuis_embedding = nn.Embedding(len(self.train_dataset.vocab_cuisine), self.cuis_embed)
        self.cuisine_encoder = nn.Sequential(
            nn.Embedding(len(self.train_dataset.vocab_cuisine), self.cuis_embed),
            nn.Linear(self.cuis_embed, 2*self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )
        
        self.cuisine_optimizer = optim.Adam(
            self.cuisine_encoder.parameters(), lr=args.learning_rate)
        self.optim_list.append(self.cuisine_optimizer)
        
        self.encoder_fusion = nn.Sequential(nn.Linear((2 if self.args.bidirectional else 1)* 2 * args.hidden_size, args.hidden_size),
                                            nn.Tanh())
        self.fusion_optim = optim.Adam(
            self.encoder_fusion.parameters(), lr=args.learning_rate)
        self.optim_list.append(self.fusion_optim)

    def forward(self, batch, iter=iter):
        """
        input_tensor: (batch_size,max_ingr)
        target_tensor: (batch_size,max_len)

        encoder_outputs (max_ingr, N, hidden_size)
        encoder_hidden (num_layers, N, hidden_size*2)
        
        return:
        decoder_outputs: (batch,max_len,size voc)
        decoder_words: final (<max_len,batch)
        decoder_attentions: (max_len,batch,max_ingr)
        """
        input_tensor = batch["ingr"].to(self.device)
        decoder_input, decoded_words, decoder_outputs, decoder_attentions,comp_ingrs,focused_ingrs = self.initForward(
            input_tensor, pairing=True)

        try:
            target_tensor = batch["target_instr"].to(self.device)
        except AttributeError:
            Warning("Evaluation mode: only taking ingredient list as input")

        encoder_outputs, encoder_hidden = self.encoder.forward_all(
            input_tensor)
        
        cuisine_tensor = batch["cuisine"].to(self.device)
        cuisine_encoding = self.cuisine_encoder(cuisine_tensor)
        cuisine_encoding = torch.stack([cuisine_encoding] * self.encoder.gru_layers)

        decoder_hidden = torch.cat(
            (encoder_hidden, cuisine_encoding), dim=2)
        decoder_hidden = self.encoder_fusion(decoder_hidden)

        sampling_proba = self.getSamplingProba(iter)

        for di in range(self.max_length):
            decoder_attentions, decoder_hidden, topi,comp_ingr  = self.forwardDecoderStep(
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

        return decoder_outputs, decoded_words, {"attentions": decoder_attentions[:di + 1],
                                                "comp_ingrs":comp_ingrs[:di+1],
                                                "focused_ingr":focused_ingrs[:di+1]}
