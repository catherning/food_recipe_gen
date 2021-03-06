from recipe_gen.seq2seq_utils import *
from recipe_gen.network import *
from recipe_gen.seq2seq_model import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

sys.path.insert(0, os.getcwd())


class HierDecoderRNN(DecoderRNN):
    def __init__(self, args, output_size, idx2word, EOS_TOK=2, DOT_TOK=13):
        super().__init__(args, output_size)
        self.args = args
        self.device = args.device
        self.idx2word = idx2word
        self.EOS_TOK = EOS_TOK
        self.DOT_TOK = DOT_TOK

        # normal GRU is the sentence RNN
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,num_layers=self.gru_layers,dropout=args.dropout)
        self.sub_gru = nn.GRU(self.word_embed, self.hidden_size,num_layers=self.gru_layers,dropout=args.dropout)

    def forward(self, decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, encoder_outputs, decoded_words, cur_step, target_tensor, sampling_proba):
        """
        input (1,batch)
        hidden (2, batch, hidden_size)
        encoder_output (max_ingr,batch, 2*hidden_size) for attention
        """
        self.batch_size = decoder_input.shape[1]
        # (1,batch,hidden)
        output = F.relu(decoder_input)
        output, decoder_hidden = self.gru(output, decoder_hidden)

        if self.gru_layers==1:
            hidden_sub = output
        elif self.gru_layers==2:
            hidden_sub = torch.cat([output,output],dim=0)
            
        for i in range(self.args.max_length):
            output_sub = self.embedding(sub_decoder_input).view(
                1, self.batch_size, -1)  # (1,batch,hidden)
            output_sub = F.relu(output_sub)

            output_sub, hidden_sub = self.sub_gru(output_sub, hidden_sub)
            output_sub = self.out(output_sub[0])

            topi = samplek(self,output_sub, decoded_words,self.idx2word, cur_step)
            decoder_outputs[:, cur_step, i] = output_sub

            if random.random() < sampling_proba:
                idx_end = ((topi == self.EOS_TOK) + (topi == self.DOT_TOK)).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break
                sub_decoder_input = topi.squeeze().detach().view(1,-1)  # detach from history as input
            else:
                sub_decoder_input = target_tensor[:, cur_step, i].unsqueeze(0)

        return output, decoder_hidden, None, decoded_words


class HierAttnDecoderRNN(HierDecoderRNN):
    def __init__(self, args, output_size, idx2word, EOS_TOK=2, DOT_TOK=13):
        super().__init__(args, output_size, idx2word, EOS_TOK, DOT_TOK)
        self.attention = Attention(args)

    def forward(self, decoder_input, sub_decoder_input, decoder_hidden, sub_decoder_outputs, encoder_outputs, decoded_words, cur_step, target_tensor, sampling_proba):
        """
        input (1,batch)
        hidden (2, batch, hidden_size)
        encoder_output (max_ingr,batch, 2*hidden_size) for attention
        
        returns:
        output
        decoder_hidden
        all_att_weights 
        decoded_words
        """
        self.batch_size = decoder_input.shape[1]

        # Main sentence GRU creates the sentence representation
        output = F.relu(decoder_input) # (1,batch,hidden)
        output, decoder_hidden = self.gru(output, decoder_hidden)

        if self.gru_layers==1:
            hidden_sub = output
        elif self.gru_layers==2:
            hidden_sub = torch.cat([output,output],dim=0)

        all_att_weights = torch.Tensor(
            self.args.max_length, self.batch_size, self.args.max_ingr)

        # Word GRU generates the sentence word by word
        for i in range(self.args.max_length):
            output_sub = self.embedding(sub_decoder_input).view(
                1, self.batch_size, -1)  # (1,batch,hidden)
            output_sub = F.relu(output_sub)

            output_sub, attn_weights = self.attention(
                output_sub, hidden_sub, encoder_outputs)
            all_att_weights[i] = attn_weights

            output_sub, hidden_sub = self.sub_gru(output_sub, hidden_sub)
            output_sub = self.out(output_sub[0])

            topi = samplek(self, output_sub, decoded_words, self.idx2word, cur_step)
            sub_decoder_outputs[:, cur_step, i] = output_sub

            if random.random() < sampling_proba:
                idx_end = ((topi == self.EOS_TOK) + (topi == self.DOT_TOK)).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break
                sub_decoder_input = topi.squeeze().detach().view(1,-1)  # detach from history as input
            else:
                sub_decoder_input = target_tensor[:, cur_step, i].unsqueeze(0)

        return output, decoder_hidden, all_att_weights, decoded_words


class HierIngrAttnDecoderRNN(HierAttnDecoderRNN):
    def __init__(self, args, output_size, idx2word, EOS_TOK=2, DOT_TOK=13):
        super().__init__(args, output_size, idx2word, EOS_TOK, DOT_TOK)
        hidden_size = args.hidden_size
        # self.gru = nn.GRU(3*hidden_size, hidden_size)
        self.attention = IngrAtt(args)
        self.sub_gru = nn.GRU(self.args.word_embed + 2 * self.hidden_size, self.hidden_size,num_layers=self.gru_layers,dropout=args.dropout)



class HierPairAttnDecoderRNN(HierDecoderRNN):
    def __init__(self, args, output_size, idx2word, EOS_TOK=2, UNK_TOK=3, DOT_TOK=13):
        super().__init__(args, output_size, idx2word, EOS_TOK, DOT_TOK)
        self.attention = IngrAtt(args)
        self.pairAttention = PairingAtt(args, unk_token=UNK_TOK)
        self.lin = nn.Linear(2*args.word_embed + args.ingr_embed + 2 * self.hidden_size, args.word_embed + 2* self.hidden_size)
        self.sub_gru = nn.GRU(self.args.word_embed + 2 * self.hidden_size, self.hidden_size,num_layers=self.gru_layers,dropout=args.dropout)

    def forward(self, decoder_input, sub_decoder_input, decoder_hidden, sub_decoder_outputs, 
            input_tensor, encoder_embedding, encoder_outputs, decoded_words, cur_step, target_tensor, sampling_proba):
        """
        input (1,batch)
        hidden (2, batch, hidden_size)
        encoder_output (max_ingr,batch, 2*hidden_size) for attention
        """
        self.batch_size = batch_size = decoder_input.shape[1]
        # (1,batch,hidden)
        output = F.relu(decoder_input)
        output, decoder_hidden = self.gru(output, decoder_hidden)

        if self.gru_layers==1:
            hidden_sub = output
        elif self.gru_layers==2:
            hidden_sub = torch.cat([output,output],dim=0)
            
        all_attn_weights = torch.zeros(self.args.max_length,batch_size,self.pairAttention.pairings.top_k)
        all_comp_ingr = torch.zeros(self.args.max_length,batch_size,self.pairAttention.pairings.top_k)
        all_ingr_id = torch.zeros(self.args.max_length,batch_size)

        for di in range(self.args.max_length):
            embedded = self.embedding(sub_decoder_input).view(
                1, self.batch_size, -1)  # (1,batch,hidden)
            embedded = F.relu(embedded)

            output_sub, attn_weights = self.attention(
                embedded, hidden_sub, encoder_outputs)

            # Selecting the focused ingredient from input then attend on compatible ingredients
            ingr_arg = torch.argmax(attn_weights, 1)
            ingr_id = torch.LongTensor(batch_size)
            for j, id in enumerate(ingr_arg):
                ingr_id[j] = input_tensor[j, id]

            out, attn_scores, comp_ingr = self.pairAttention(
                embedded, hidden_sub, ingr_id, encoder_embedding) 
            # changed embedding (input of Main dec) to output_sub (like IngrAtt above)
            all_attn_weights[di] = attn_scores
            all_comp_ingr[di] = comp_ingr
            all_ingr_id[di] = ingr_id
                
            if out is not None:
                output_sub = torch.cat((output_sub,out),dim=-1)#self.lin(output)
                output_sub = self.lin(output_sub)

            output_sub, hidden_sub = self.sub_gru(output_sub, hidden_sub)
            output_sub = self.out(output_sub[0])

            topi = samplek(self, output_sub, decoded_words, self.idx2word, cur_step)
            sub_decoder_outputs[:, cur_step, di] = output_sub

            if random.random() < sampling_proba:
                idx_end = ((topi == self.EOS_TOK) + (topi == self.DOT_TOK)).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break
                sub_decoder_input = topi.squeeze().detach().view(1,-1)  # detach from history as input
            else:
                sub_decoder_input = target_tensor[:, cur_step, di].unsqueeze(0)

        return output, decoder_hidden, all_attn_weights, decoded_words, all_comp_ingr, all_ingr_id
