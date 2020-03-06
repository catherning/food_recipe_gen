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
        self.sub_gru = nn.GRU(self.hidden_size, self.hidden_size)  # word RNN

    def samplek(self, decoder_output, decoded_words, cur_step):
        # TODO: change for hierarchical
        topv, topi = decoder_output.topk(self.args.topk)
        distrib = torch.distributions.categorical.Categorical(logits=topv)
        chosen_id = torch.zeros(
            decoder_output.shape[0], dtype=torch.long, device=self.device)
        for batch_id, idx in enumerate(distrib.sample()):
            chosen_id[batch_id] = topi[batch_id, idx]
            decoded_words[batch_id][cur_step].append(
                self.idx2word[chosen_id[batch_id].item()])
        return chosen_id

    def forward(self, decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, encoder_outputs, decoded_words, cur_step, target_tensor, sampling_proba):
        """
        input (1,batch)
        hidden (2, batch, hidden_size)
        encoder_output (max_ingr,batch, 2*hidden_size) for attention
        """
        self.batch_size = decoder_input.shape[1]
        output = self.embedding(decoder_input).view(
            1, self.batch_size, -1)  # (1,batch,hidden)
        output = F.relu(output)

        output, decoder_hidden = self.gru(output, decoder_hidden)

        hidden_sub = output
        # XXX: is it ok same embedding for both dec and subdec input ??? for now the init inputs are the same

        for i in range(self.args.max_length):
            output_sub = self.embedding(sub_decoder_input).view(
                1, self.batch_size, -1)  # (1,batch,hidden)
            output_sub = F.relu(output_sub)

            output_sub, hidden_sub = self.sub_gru(output_sub, hidden_sub)
            output_sub = self.softmax(self.out(output_sub[0]))

            topi = self.samplek(output_sub, decoded_words, cur_step)
            decoder_outputs[:, cur_step, i] = output_sub

            if random.random() < sampling_proba:
                idx_end = ((topi == self.EOS_TOK) + (topi == self.DOT_TOK)).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break
                output_sub = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input
            else:
                output_sub = target_tensor[:, cur_step, i].view(1, -1)

        return output_sub, decoder_hidden, None, decoded_words


class HierAttnDecoderRNN(HierDecoderRNN):
    def __init__(self, args, output_size, idx2word, EOS_TOK=2, DOT_TOK=13):
        super().__init__(args, output_size, idx2word, EOS_TOK, DOT_TOK)
        self.attention = Attention(args)

    def forward(self, decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, encoder_outputs, decoded_words, cur_step, target_tensor, sampling_proba):
        """
        input (1,batch)
        hidden (2, batch, hidden_size)
        encoder_output (max_ingr,batch, 2*hidden_size) for attention
        """
        self.batch_size = decoder_input.shape[1]
        output = self.embedding(decoder_input).view(
            1, self.batch_size, -1)  # (1,batch,hidden)
        output = F.relu(output)

        output, decoder_hidden = self.gru(output, decoder_hidden)

        hidden_sub = output
        # XXX: is it ok same embedding for both dec and subdec input ??? for now the init inputs are the same

        all_att_weights = torch.Tensor(
            self.args.max_length, self.batch_size, self.args.max_ingr)

        for i in range(self.args.max_length):
            output_sub = self.embedding(sub_decoder_input).view(
                1, self.batch_size, -1)  # (1,batch,hidden)
            output_sub = F.relu(output_sub)

            output_sub, attn_weights = self.attention(
                output_sub, hidden_sub, encoder_outputs)
            all_att_weights[i] = attn_weights

            output_sub, hidden_sub = self.sub_gru(output_sub, hidden_sub)
            output_sub = self.softmax(self.out(output_sub[0]))

            topi = self.samplek(output_sub, decoded_words, cur_step)
            decoder_outputs[:, cur_step, i] = output_sub

            if random.random() < sampling_proba:
                idx_end = ((topi == self.EOS_TOK) + (topi == self.DOT_TOK)).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break
                output_sub = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input
            else:
                output_sub = target_tensor[:, cur_step, i].view(1, -1)

        return output_sub, decoder_hidden, all_att_weights, decoded_words


class HierIngrAttnDecoderRNN(HierAttnDecoderRNN):
    def __init__(self, args, output_size, idx2word, EOS_TOK=2, DOT_TOK=13):
        super().__init__(args, output_size, idx2word, EOS_TOK, DOT_TOK)
        hidden_size = args.hidden_size
        # self.gru = nn.GRU(3*hidden_size, hidden_size)
        self.attention = IngrAtt(args)


class HierPairAttnDecoderRNN(HierDecoderRNN):
    def __init__(self, args, output_size, idx2word, EOS_TOK=2, UNK_TOK=3, DOT_TOK=13):
        super().__init__(args, output_size, idx2word, EOS_TOK, DOT_TOK)
        self.attention = IngrAtt(args)
        self.pairAttention = PairingAtt(args, unk_token=UNK_TOK)
        # self.gru = nn.GRU(2*args.hidden_size, args.hidden_size)
        self.lin = nn.Linear(3*args.hidden_size, 2*args.hidden_size)

        # self.gru = nn.GRU(2*args.hidden_size, args.hidden_size)

    def forward(self, decoder_input, sub_decoder_input, decoder_hidden, decoder_outputs, 
            input_tensor, encoder_embedding, encoder_outputs, decoded_words, cur_step, target_tensor, sampling_proba):
        """
        input (1,batch)
        hidden (2, batch, hidden_size)
        encoder_output (max_ingr,batch, 2*hidden_size) for attention
        """
        self.batch_size = decoder_input.shape[1]
        embedded = self.embedding(decoder_input).view(
            1, self.batch_size, -1)  # (1,batch,hidden)
        output = F.relu(embedded)

        output, decoder_hidden = self.gru(output, decoder_hidden)

        hidden_sub = output
        # XXX: is it ok same embedding for both dec and subdec input ??? for now the init inputs are the same

        for i in range(self.args.max_length):
            output_sub = self.embedding(sub_decoder_input).view(
                1, self.batch_size, -1)  # (1,batch,hidden)
            output_sub = F.relu(output_sub)

            output_sub, attn_weights = self.attention(
                output_sub, hidden_sub, encoder_outputs)

            # Selecting the focused ingredient from input then attend on compatible ingredients
            ingr_arg = torch.argmax(attn_weights, 1)
            ingr_id = torch.LongTensor(batch_size)
            for i, id in enumerate(ingr_arg):
                ingr_id[i] = input_tensor[i, id]

            out, attn_weights = self.pairAttention(
                embedded, hidden_sub, ingr_id, encoder_embedding)
            if out is None:
                output = self.lin(output)
            else:
                output = out

            output_sub, hidden_sub = self.sub_gru(output_sub, hidden_sub)
            output_sub = self.softmax(self.out(output_sub[0]))

            topi = self.samplek(output_sub, decoded_words, cur_step)
            decoder_outputs[:, cur_step, i] = output_sub

            if random.random() < sampling_proba:
                idx_end = ((topi == self.EOS_TOK) + (topi == self.DOT_TOK)).nonzero()[
                    :, 0]
                if len(idx_end) == self.batch_size:
                    break
                output_sub = topi.squeeze().detach().view(
                    1, -1)  # detach from history as input
            else:
                output_sub = target_tensor[:, cur_step, i].view(1, -1)

        return output_sub, decoder_hidden, attn_weights, decoded_words
