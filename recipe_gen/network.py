import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from recipe_gen.seq2seq_utils import MAX_INGR, MAX_LENGTH
from recipe_gen.pairing_utils import PairingData


class EncoderRNN(nn.Module):
    def __init__(self, args, input_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.device = args.device
        self.max_ingr = args.max_ingr
        self.batch_size = args.batch_size

        self.embedding = nn.Embedding(input_size, args.hidden_size)
        self.gru = nn.GRU(args.hidden_size, args.hidden_size, bidirectional=True) # TODO: change size as biGRU => num_dir = 2

    def forward(self, input_, hidden):
        """
        input_: (batch)
        hidden: (1,batch,hidden * 2) *2 bc biGRU
        output: (1,batch,hidden * 2) 
        """
        embedded = self.embedding(input_).view(1, self.batch_size, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, self.batch_size, self.hidden_size) # (num_layers * num_directions, batch, hidden_size)

    def forward_all(self, input_tensor):
        """
        input: (batch,max_ingr) ?
        encoder_outputs: (max_ingr,batch,hidden*2)
        encoder_hidden: (2,batch,hidden)
        """
        self.batch_size = len(input_tensor)
        encoder_hidden = self.initHidden().to(self.device)
        encoder_outputs = torch.zeros(
            self.max_ingr, self.batch_size, self.hidden_size*2, device=self.device)

        for ei in range(self.max_ingr):
            # TODO: couldn't give directly all input_tensor, not step by step ???
            encoder_output, encoder_hidden = self.forward(
                input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        return encoder_outputs, encoder_hidden

class DecoderRNN(nn.Module):
    def __init__(self, args, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.batch_size = args.batch_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.hiddenLayer = nn.Linear(2*hidden_size,hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_output):
        """
        input (1,batch)
        hidden (2, batch, hidden_size)
        encoder_output (max_ingr,batch, 2*hidden_size) 
        """
        self.batch_size = input.shape[1]
        output = self.embedding(input).view(1, self.batch_size, -1) #(1,batch,hidden)
        output = F.relu(output)

        hidden = torch.cat((hidden[0],hidden[1]),1).unsqueeze(0)
        hidden = self.hiddenLayer(hidden) # because was size 2-hidden_size
        
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, None

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class HierDecoderRNN(nn.Module):
    def __init__(self, args, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.batch_size = args.batch_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.hiddenLayer = nn.Linear(2*hidden_size,hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size) #sentence RNN
        self.sub_gru = nn.GRU(hidden_size, hidden_size) #word RNN
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_output):
        """
        input (1,batch)
        hidden (2, batch, hidden_size)
        encoder_output (max_ingr,batch, 2*hidden_size) 
        """
        self.batch_size = input.shape[1]
        output = self.embedding(input).view(1, self.batch_size, -1) #(1,batch,hidden)
        output = F.relu(output)

        hidden = torch.cat((hidden[0],hidden[1]),1).unsqueeze(0)
        hidden = self.hiddenLayer(hidden) # because was size 2-hidden_size
        
        for 

        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, None

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(DecoderRNN):
    def __init__(self, args, output_size):
        super().__init__(args, output_size)
        self.attention = Attention(args)

    def forward(self, input, hidden, encoder_outputs):
        """
        input: (1,batch)
        hidden: (1,batch,hidden)
        encoder_outputs: (max_ingr,batch,2*hidden)
        """
        self.batch_size = input.shape[1]
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        # embedded (1,batch,2*hidden)

        hidden = torch.cat((hidden[0],hidden[1]),1).unsqueeze(0)
        hidden = self.hiddenLayer(hidden) # because was size 2-hidden_size

        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights


class IngrAttnDecoderRNN(DecoderRNN):
    def __init__(self, args, output_size):
        super().__init__(args, output_size)
        hidden_size = args.hidden_size
        self.gru = nn.GRU(3*hidden_size, hidden_size)
        self.attention = IngrAtt(args)

    def forward(self, input, hidden, encoder_outputs):
        """
        input: (1,batch)
        hidden: (1,batch,hidden)
        encoder_outputs: (max_ingr,hidden)
        """
        self.batch_size = input.shape[1]
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        # embedded (1,batch,hidden) ?

        hidden = torch.cat((hidden[0],hidden[1]),1).unsqueeze(0)
        hidden = self.hiddenLayer(hidden) # because was size 2-hidden_size

        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights


class PairAttnDecoderRNN(AttnDecoderRNN):
    def __init__(self, args, output_size, unk_token=3):
        super().__init__(args, output_size)
        self.attention = IngrAtt(args)
        self.pairAttention = PairingAtt(args, unk_token=unk_token)
        self.gru = nn.GRU(2*args.hidden_size, args.hidden_size)
        self.lin = nn.Linear(3*args.hidden_size,2*args.hidden_size)

        if args.model_name=="Seq2seqTitlePairing":
            self.hiddenLayer = nn.Linear(4*args.hidden_size,args.hidden_size)
        else:
            self.hiddenLayer = nn.Linear(2*args.hidden_size,args.hidden_size)
            # self.gru = nn.GRU(2*args.hidden_size, args.hidden_size)
    
    def forward(self, input, hidden, encoder_outputs, encoder_embedding, input_tensor):
        """
        input: (1,batch)
        hidden: (1,batch,hidden)
        encoder_outputs: (max_ingr,hidden)

        output
        """
        batch_size = hidden.shape[1]
        embedded = self.embedding(input)  # (1, batch_size, hidden)

        hidden = torch.cat((hidden[0],hidden[1]),1).unsqueeze(0)
        hidden = self.hiddenLayer(hidden) # because was size 2-hidden_size

        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        # Selecting the focused ingredient from input then attend on compatible ingredients
        ingr_arg = torch.argmax(attn_weights, 1)
        ingr_id = torch.LongTensor(batch_size)
        for i, id in enumerate(ingr_arg):
            ingr_id[i] = input_tensor[i, id]

        out, attn_weights = self.pairAttention(
            embedded, hidden, ingr_id, encoder_embedding)
        if out is None:
            output = self.lin(output)
        else:
            output = out

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout_p = args.dropout
        self.max_ingr = args.max_ingr
        self.max_length = args.max_length
        self.hidden_size = args.hidden_size

        self.dropout = nn.Dropout(self.dropout_p)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_ingr)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)

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
        attn_applied = torch.bmm(attn_weights.view(-1, 1, self.max_ingr),
                                 encoder_outputs.view(-1, self.max_ingr, 2*self.hidden_size))
        # attn_applied: (batch,1,2*hidden)

        output = torch.cat((embedded[0], attn_applied[:, 0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        return output, attn_weights


class IngrAtt(Attention):
    def __init__(self, args):
        super().__init__(args)
        self.attn = nn.Linear(self.hidden_size * 3, self.max_ingr)

    def forward(self, embedded, hidden, encoder_outputs):
        """Def from user pref paper
        K: encoder_outputs (max_ingr,batch,hidden)
        Q: hidden (1,batch,hidden)
        V: encoder_outputs
        """
        batch_size = embedded.shape[1]
        attn_weights = F.softmax(torch.tanh(self.attn(
                        torch.cat((encoder_outputs[-1], hidden[0]), 1)
                        )),dim=1)
        
        # attn_weights view: (batch,1,max_ingr)
        # encoder_outputs view: (batch,max_ingr,hidden_size)
        attn_applied = torch.bmm(attn_weights.view(batch_size, 1, self.max_ingr),
                                 encoder_outputs.view(batch_size, self.max_ingr, 2*self.hidden_size))
        output = torch.cat((embedded[0], attn_applied[:, 0]), 1).unsqueeze(0)

        return output, attn_weights


class PairingAtt(Attention):
    def __init__(self, args, unk_token=3):
        super().__init__(args)
        with open(args.pairing_path, 'rb') as f:
            self.pairings = pickle.load(f)
            
        self.unk_token = unk_token
        self.attn = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, embedded, hidden, ingr_id, encoder_embedding):
        """
        K: ing_j that needs to be retrieved
        Q: h_enc,t = i_enc,j max from previous attention
        V: ing_j retrieved

        returns
        output: (1,batch,2*hidden)
        attn_scores: (batch,top_k) 
        """

        compatible_ingr = [self.pairings.bestPairingsFromIngr(
            ingr) for ingr in ingr_id]

        batch_size = embedded.shape[1]
        scores = torch.zeros(batch_size, self.pairings.top_k)
        comp_ingr_id = torch.ones(
            batch_size, self.pairings.top_k, dtype=torch.long,device=embedded.device)*self.unk_token
        for i, batch in enumerate(compatible_ingr):
            # if len(batch) > 0: # necessary ?
            comp_ingr = []
            for j,pair in enumerate(batch):
                scores[i, j] = pair[1]
                comp_ingr.append(pair[0])

            comp_ingr = torch.LongTensor(comp_ingr)
            comp_ingr_id[i, :comp_ingr.shape[0]] = comp_ingr

        comp_emb = encoder_embedding(comp_ingr_id)

        attn_weights = torch.zeros(batch_size, self.pairings.top_k)
        for i in range(self.pairings.top_k):
            attn_weights[:, i] = self.attn(
                torch.cat((comp_emb[:, i], hidden[0]), 1)
                
                ).view(batch_size)
        attn_weights = F.softmax(torch.tanh(attn_weights), dim=1)

        # XXX: renormalize after multiplication ?
        # TODO: try with emphazing unknown pairings
        attn_scores = (attn_weights * scores).to(comp_emb.device)

        # TODO: try with still doing the attention, but without the scores if there's no compatible ingr ?
        # or too broad to add ingr afterwards ?
        if scores.sum() == 0:
            return None, attn_scores

        # attn_scores view: (batch,1,top_k)
        # comb_emb (batch,top_k,hidden_size)
        attn_applied = torch.bmm(attn_scores.view(-1, 1, self.pairings.top_k),
                                 comp_emb)

        output = torch.cat((embedded[0], attn_applied[:, 0]), 1).unsqueeze(0)

        return output, attn_scores
