import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from recipe_1m_analysis.utils import MAX_INGR, MAX_LENGTH
import recipe_gen.pairing_utils as pairing


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, max_ingr=MAX_INGR, device="cpu"):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.max_ingr = max_ingr
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_, hidden):
        embedded = self.embedding(input_).view(1, self.batch_size, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size)

    def forward_all(self, input_tensor):
        encoder_hidden = self.initHidden().to(self.device)
        encoder_outputs = torch.zeros(  # self.batch_size, #XXX: ??
            self.max_ingr, self.hidden_size, device=self.device)

        for ei in range(self.max_ingr):
            # TODO: couldn't give directly all input_tensor, not step by step ???
            encoder_output, encoder_hidden = self.forward(
                input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        return encoder_outputs, encoder_hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

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
    def __init__(self, hidden_size, output_size, batch_size, dropout_p=0.1, max_ingr=MAX_INGR, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__(
            hidden_size, output_size, batch_size)
        self.attention = Attention(
            hidden_size, dropout_p=dropout_p, max_ingr=max_ingr, max_length=max_length)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, self.batch_size, -1)

        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights


class Attention(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1, max_ingr=MAX_INGR, max_length=MAX_LENGTH):
        super().__init__()
        self.dropout_p = dropout_p
        self.max_ingr = max_ingr
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size * 2, self.max_ingr)
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

class PairingAtt(Attention):
    def __init__(self,filepath):
        super().__init__()
        self.pairings = pairing.PairingData(filepath)

    def forward(self,):
        pass