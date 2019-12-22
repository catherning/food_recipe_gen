import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from recipe_gen.seq2seq_utils import MAX_INGR, MAX_LENGTH
from recipe_gen.pairing_utils import PairingData


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
        """
        input_: (batch)
        hidden: (1,batch,hidden)
        output: (1,batch,hidden)
        """
        embedded = self.embedding(input_).view(1, self.batch_size, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size)

    def forward_all(self, input_tensor):
        """
        input: (batch,max_ingr) ?
        encoder_outputs: (max_ingr,batch,hidden)
        encoder_hidden: (1,batch,hidden)
        """
        self.batch_size = len(input_tensor)
        encoder_hidden = self.initHidden().to(self.device)
        encoder_outputs = torch.zeros(
            self.max_ingr, self.batch_size,self.hidden_size, device=self.device)

        for ei in range(self.max_ingr):
            # TODO: couldn't give directly all input_tensor, not step by step ???
            encoder_output, encoder_hidden = self.forward(
                input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

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

    def forward(self, input, hidden,encoder_output):
        self.batch_size = input.shape[1]
        output = self.embedding(input).view(1, self.batch_size, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden,None

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(DecoderRNN):
    def __init__(self, hidden_size, output_size, batch_size, dropout_p=0.1, max_ingr=MAX_INGR, max_length=MAX_LENGTH):
        super().__init__(
            hidden_size, output_size, batch_size)
        self.attention = Attention(
            hidden_size, dropout_p=dropout_p, max_ingr=max_ingr, max_length=max_length)

    def forward(self, input, hidden, encoder_outputs):
        """
        input: (1,batch)
        hidden: (1,batch,hidden)
        encoder_outputs: (max_ingr,hidden)
        """
        self.batch_size = input.shape[1]
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        # embedded (1,batch,hidden) ?

        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights

class IngrAttnDecoderRNN(DecoderRNN):
    def __init__(self, hidden_size, output_size, batch_size, dropout_p=0.1, max_ingr=MAX_INGR, max_length=MAX_LENGTH):
        super().__init__(hidden_size, output_size, batch_size)
        self.gru = nn.GRU(2*hidden_size, hidden_size)
        self.attention = IngrAtt(
            hidden_size, dropout_p=dropout_p, max_ingr=max_ingr, max_length=max_length)

    def forward(self, input, hidden, encoder_outputs):
        """
        input: (1,batch)
        hidden: (1,batch,hidden)
        encoder_outputs: (max_ingr,hidden)
        """
        self.batch_size = input.shape[1]
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        # embedded (1,batch,hidden) ?

        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights


class PairAttnDecoderRNN(AttnDecoderRNN):
    def __init__(self, filepath,hidden_size, output_size, batch_size, dropout_p=0.1, max_ingr=MAX_INGR, max_length=MAX_LENGTH,unk_token=3):
        super().__init__(
            hidden_size, output_size, batch_size)
        self.attention = IngrAtt(
            hidden_size, dropout_p=dropout_p, max_ingr=max_ingr, max_length=max_length)
        self.pairAttention = PairingAtt(filepath,hidden_size, dropout_p=dropout_p, max_ingr=max_ingr, max_length=max_length,unk_token=unk_token)
        self.gru = nn.GRU(2*hidden_size, hidden_size)

    def forward(self, input, hidden, encoder_outputs,encoder_embedding,input_tensor):
        """
        input: (1,batch)
        hidden: (1,batch,hidden)
        encoder_outputs: (max_ingr,hidden)
        """
        embedded = self.embedding(input).view(1, self.batch_size, -1)

        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        ingr_arg = torch.argmax(attn_weights,1)
        # ingr_arg = torch.LongTensor([[i,id] for i,id in enumerate(ingr_arg)]).to(ingr_arg.device)
        # ingr_id = input_tensor[ingr_arg[0]]
        ingr_id = torch.LongTensor(self.batch_size)
        for i,id in enumerate(ingr_arg):
            ingr_id[i]=input_tensor[i,id]

        out,attn_weights = self.pairAttention(embedded,hidden,ingr_id,encoder_embedding)
        if out is not None:
            output = out
        
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

        self.dropout = nn.Dropout(self.dropout_p)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_ingr)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, embedded, hidden, encoder_outputs):
        """From pytorch seq2seq tuto
        key: encoder_outputs
        query: embedded
        value: hidden

        or ?
        K: embedded (1,batch,hidden)
        Q: hidden (1,batch,hidden)
        V: encoder_outputs (max_ingr,hidden) now (max_ingr,batch_size,hidden)

        returns:
        output: (1,batch,hidden)
        attn_weights (batch,max_ingr)

        """
        embedded = self.dropout(embedded)

        # attn_weights : (batch,max_ingr)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.view(-1,1,self.max_ingr),
                                 encoder_outputs.view(-1,self.max_ingr,self.hidden_size))
        # attn_applied: (batch,1,hidden)

        output = torch.cat((embedded[0], attn_applied[:,0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        return output, attn_weights

class IngrAtt(Attention):
    def __init__(self, hidden_size, dropout_p=0.1, max_ingr=MAX_INGR, max_length=MAX_LENGTH):
        super().__init__(hidden_size, dropout_p=dropout_p, max_ingr=max_ingr, max_length=max_length)
        # self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self,embedded,hidden,encoder_outputs):
        """Def from user pref paper
        K: encoder_outputs (max_ingr,batch,hidden)
        Q: hidden (1,batch,hidden)
        V: encoder_outputs
        """

        # attn_weights = torch.zeros(embedded.shape[1],self.max_ingr)
        # for i in range(self.max_ingr):
        #     attn_weights[:,i]=F.softmax(torch.tanh(
        #     self.attn(torch.cat((encoder_outputs[i], hidden[0]), 1))
        #     ), dim=1)

        attn_weights = F.softmax(torch.tanh(
            self.attn(torch.cat((encoder_outputs[-1], hidden[0]), 1))
            ), dim=1)

        # attn_weights view: (batch,1,max_ingr)
        # encoder_outputs view: (batch,max_ingr,hidden_size)
        attn_applied = torch.bmm(attn_weights.view(-1,1,self.max_ingr),
            encoder_outputs.view(-1,self.max_ingr,self.hidden_size))
        output = torch.cat((embedded[0], attn_applied[:,0]), 1).unsqueeze(0)
        
        # attn_weights (max_ingr,batch,max_ingr)
        # attn_weights = F.softmax(torch.tanh(
        #     self.attn(torch.cat((encoder_outputs, hidden.repeat(10,1,1)), 2))
        #     ), dim=1)
        # attn_applied = torch.bmm(attn_weights.view(-1,self.max_ingr,self.max_ingr),
        #                          encoder_outputs.view(-1,self.max_ingr,self.hidden_size))

        # output = torch.cat((embedded.repeat(10,1,1), attn_applied.view(self.max_ingr,-1,self.hidden_size)), 2)
        # similar to classic attention, but no attn_combine layer in the paper before putting in the GRU ?
        # in the paper, use BiGRU, so normal to have 2*hidden_size, but need to redefine GRU then
        return output, attn_weights

class PairingAtt(Attention):
    def __init__(self,filepath, hidden_size, dropout_p=0.1, max_ingr=MAX_INGR, max_length=MAX_LENGTH,unk_token=3):
        super().__init__(hidden_size, dropout_p=dropout_p, max_ingr=max_ingr, max_length=max_length)
        with open(filepath,'rb') as f:
            self.pairings = pickle.load(f)
        # self.pairings = PairingData(filepath)
        self.unk_token = unk_token
        self.attn = nn.Linear(self.hidden_size * 2, 1)

    def forward(self,embedded,hidden,ingr_id,encoder_embedding):
        """
        K: ing_j that needs to be retrieved
        Q: h_enc,t = i_enc,j max from previous attention
        V: ing_j retrieved
        """

        compatible_ingr = [self.pairings.bestPairingsFromIngr(ingr) for ingr in ingr_id]
        
        batch_size = embedded.shape[1]
        comp_ingr_id = torch.ones(batch_size,self.pairings.top_k,dtype=torch.int)*self.unk_token
        for i in range(batch_size):
            if len(compatible_ingr[i])>0:
                comp_ingr = torch.LongTensor([pair[0] for pair in compatible_ingr[i]])
                comp_ingr_id[i,:comp_ingr.shape[0]] = comp_ingr

        comp_emb = encoder_embedding(comp_ingr_id.to(embedded.device).long())

        attn_weights = torch.zeros(embedded.shape[1],self.pairings.top_k)
        for i in range(self.pairings.top_k):
            attn_weights[:,i]=self.attn(torch.cat((comp_emb[:,i], hidden[0]), 1)).view(embedded.shape[1])
        attn_weights = F.softmax(torch.tanh(attn_weights),dim=1)

        # attn_weights = F.softmax(torch.tanh(
        #     self.attn(torch.cat((comp_emb, hidden.repeat(self.pairings.top_k,1,1).view(-1,self.pairings.top_k,self.hidden_size)), 2))), dim=1)
        
        # TODO: take at the same time as the selection of ingr_id in comp_ingr_id ?
        scores = torch.zeros(embedded.shape[1],self.pairings.top_k)
        for i,batch in enumerate(compatible_ingr):
            for j,pair in enumerate(batch):
                scores[i,j]=pair[1]
        # scores = torch.Tensor([[pair[1]] for batch in compatible_ingr for pair in batch])
        
        # XXX: renormalize after multiplication ?
        # TODO: try with emphazing unknown pairings
        attn_scores = (attn_weights * scores).to(comp_emb.device)

        # TODO: try with still doing the attention, but without the scores if there's no compatible ingr ?
        # or too broad to add ingr afterwards ?
        if scores.sum()==0:
            return None,attn_scores

        # attn_scores view: (batch,1,top_k)
        # comb_emb (batch,top_k,hidden_size)
        attn_applied = torch.bmm(attn_scores.view(-1,1,self.pairings.top_k),
                                comp_emb)

        output = torch.cat((embedded[0], attn_applied[:,0]), 1).unsqueeze(0)
                
        return output,attn_scores