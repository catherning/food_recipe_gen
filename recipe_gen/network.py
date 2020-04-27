import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from recipe_gen.seq2seq_utils import MAX_INGR, MAX_LENGTH
from recipe_gen.pairing_utils import PairingData

class EncoderRNN(nn.Module):
    def __init__(self, args, input_size,embedding_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.device = args.device
        self.max_ingr = args.max_ingr
        self.batch_size = args.batch_size
        self.word_embed = embedding_size
        self.gru_layers = args.num_gru_layers
        self.num_directions = 2 if args.bidirectional else 1
        
        self.embedding = nn.Embedding(input_size, self.word_embed)
        self.gru = nn.GRU(self.word_embed, args.hidden_size, bidirectional=args.bidirectional,num_layers=self.gru_layers,dropout=args.dropout)

        # so that correct hidden dims for decoder because biGRU
        # self.hiddenLayer = nn.Linear(2*args.hidden_size,args.hidden_size)

    def forward(self, input_, hidden):
        """
        input_: (batch)
        hidden: ((2 if self.bidirectional else 1) * num_layers, batch, hidden) *2 bc biGRU
        
        returns
        output: (1,batch,hidden * 2)
        hidden
        """
        output = self.embedding(input_).view(1, self.batch_size, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_directions * self.gru_layers, self.batch_size, self.hidden_size) # (num_layers * num_directions, batch, hidden_size)

    def forward_all(self, input_tensor):
        """
        input: (batch,max_ingr)
        encoder_outputs: (max_ingr,batch,hidden*2)
        encoder_hidden: (2 * num_layers ,batch,hidden) -> (num_layers,batch,hidden)
        """
        #TODO: check, after permute, EOS n'est plus à la fin!?
        input_tensor=input_tensor[:,torch.randperm(input_tensor.size()[1])]
        self.batch_size = batch_size = len(input_tensor)
        encoder_hidden = self.initHidden().to(self.device)
        encoder_outputs = torch.zeros(
            self.max_ingr, batch_size, self.hidden_size*2, device=self.device)

        for ei in range(self.max_ingr):
            # TODO: couldn't give directly all input_tensor, not step by step ???
            encoder_output, encoder_hidden = self.forward(
                input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        if self.gru_layers > 1 and self.num_directions==2:
            encoder_hidden = encoder_hidden.view(self.gru_layers, self.num_directions, self.batch_size, self.hidden_size)
            encoder_hidden = torch.cat((encoder_hidden[:,0],encoder_hidden[:,1]),2)
        elif self.num_directions==2:
            encoder_hidden = torch.cat((encoder_hidden[0],encoder_hidden[1]),1).unsqueeze(0)
    
        # encoder_hidden = F.relu(self.hiddenLayer(encoder_hidden)) # because was size 2-hidden_size
        # XXX: no relu ?
        # put hiddenLayer out of encoder, fuse with encoder_fusion

        return encoder_outputs, encoder_hidden

class DecoderRNN(nn.Module):
    def __init__(self, args, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.word_embed = args.word_embed
        self.gru_layers = args.num_gru_layers

        self.embedding = nn.Embedding(output_size, self.word_embed)
        self.gru = nn.GRU(self.word_embed, hidden_size,num_layers=self.gru_layers,dropout=args.dropout)
        self.out = nn.Linear(hidden_size, output_size)
        #self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_output):
        """
        input (1,N)
        hidden (num_layers, N, hidden_size)
        encoder_output (max_ingr, N, 2*hidden_size) 
        
        returns 
        output (1, N, word_embed) -> (1, N, hidden_size) -> (N, vocab_tok_size)
        hidden (num_layers, N, hidden_size)
        """
        self.batch_size = input.shape[1]
        output = self.embedding(input).view(1, self.batch_size, -1) #(1,batch,word_embed)
        output = F.relu(output)
        
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden, None, None

 
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

        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights, None


class IngrAttnDecoderRNN(AttnDecoderRNN):
    def __init__(self, args, output_size):
        super().__init__(args, output_size)
        hidden_size = args.hidden_size
        self.gru = nn.GRU(args.word_embed + 2 * hidden_size, hidden_size,num_layers=self.gru_layers,dropout=args.dropout)
        self.attention = IngrAtt(args)

class PairAttnDecoderRNN(AttnDecoderRNN):
    def __init__(self, args, output_size, unk_token=3):
        super().__init__(args, output_size)
        hidden_size = args.hidden_size
        self.attention = IngrAtt(args)
        self.pairAttention = PairingAtt(args, unk_token=unk_token)
        self.lin = nn.Linear(args.word_embed * 2+ args.ingr_embed+ 2* hidden_size, args.word_embed + 2* hidden_size)
        self.gru = nn.GRU(args.word_embed + 2* hidden_size, hidden_size,num_layers=self.gru_layers,dropout=args.dropout)
    
    def forward(self, input, hidden, encoder_outputs, encoder_embedding, input_tensor):
        """
        input: (1,batch)
        hidden: (1,batch,hidden)
        encoder_outputs: (max_ingr,hidden)

        output
        """
        batch_size = hidden.shape[1]
        embedded = self.embedding(input)  # (1, batch_size, word_embed)

        # TODO: in trained model, check if max(attn_weight) is always at same arg...
        # output (1,N,word_embed + hidden * 2)
        output, attn_weights = self.attention(
            embedded, hidden, encoder_outputs)

        # Selecting the focused ingredient from input then attend on compatible ingredients
        #TODO: if keep being always the same, sample from weights ?, or give sth else than encoder_outputs ? encoder_hidden ?
        # Or select the first that is not eos or pad
        ingr_arg = torch.argmax(attn_weights, 1)
        ingr_id = input_tensor[torch.arange(batch_size),ingr_arg]
            
        out, attn_weights, comp_ingr = self.pairAttention(
            embedded, hidden, ingr_id, encoder_embedding)
        if out is not None:
            output = torch.cat((output,out),dim=-1)#self.lin(output)
            output = self.lin(output)

        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights, comp_ingr, ingr_id

class BaseAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_ingr = args.max_ingr
        self.max_length = args.max_length
        self.hidden_size = args.hidden_size


class Attention(BaseAttention):
    def __init__(self, args):
        super().__init__(args)
        hidden_size = self.hidden_size
        self.attn = nn.Linear(hidden_size + args.word_embed, self.max_ingr)
        self.dropout = nn.Dropout(args.dropout)

        self.attn_combine = nn.Linear(hidden_size*2 + args.word_embed, args.word_embed)

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


class IngrAtt(BaseAttention):
    def __init__(self, args):
        super().__init__(args)
        hidden_size = self.hidden_size
        self.key_layer = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn = nn.Linear(hidden_size, 1, bias= False)

    def forward(self, embedded, hidden, encoder_outputs):
        """Def from user pref paper
        Input:
        embedded (1,N,word_embed + hidden * 2)
        Q: hidden (1,batch,hidden)
        K: encoder_outputs (max_ingr,batch,hidden*2)
        V: encoder_outputs (max_ingr,N,hidden*2)
        
        returns
        output (1,N, word_embed + 2 * hidden)
        attn_weights (N, max_ingr)
        """
        batch_size = embedded.shape[1]
        hidden_repeat = hidden[0].unsqueeze(1).expand(-1,self.max_ingr,-1)
        
        query = self.query_layer(hidden_repeat)
        key = self.key_layer(encoder_outputs.view(batch_size,self.max_ingr,self.hidden_size *2))

        scores = self.attn(torch.tanh(query + key))
        attn_weights = F.softmax(scores.squeeze(2), dim=-1)
        # context = torch.bmm(alphas, value)
        
        attn_applied = torch.bmm(attn_weights.view(batch_size, 1, self.max_ingr),
                                 encoder_outputs.view(batch_size, self.max_ingr, 2*self.hidden_size))
        output = torch.cat((embedded[0], attn_applied[:, 0]), 1).unsqueeze(0)
        # attn_weights view: (batch,1,max_ingr)
        # encoder_outputs view: (batch,max_ingr,hidden_size)

        return output, attn_weights


class PairingAtt(IngrAtt):
    def __init__(self, args, unk_token=3):
        super().__init__(args)
        with open(args.pairing_path, 'rb') as f:
            self.pairings = pickle.load(f)
            
        self.unk_token = unk_token
        self.key_layer = nn.Linear(args.ingr_embed, self.hidden_size, bias=False)

    def forward(self, embedded, hidden, ingr_id, encoder_embedding):
        """
        K: ing_j that needs to be retrieved
        Q: h_enc,t = i_enc,j max from previous attention
        V: ing_j retrieved

        returns
        output: (1,batch, word_embed + ingr_embed)
        attn_scores: (batch,top_k) 
        """
        batch_size = embedded.shape[1]
        device = embedded.device
        scores = torch.zeros(batch_size, self.pairings.top_k).to(device)
        comp_ingr_id = torch.ones(batch_size, self.pairings.top_k, dtype=torch.long, device = device)*self.unk_token
        
        for i,(comp_ingr, score_list) in enumerate(map(self.pairings.bestPairingsFromIngr, ingr_id)):
            comp_ingr_id[i, :len(comp_ingr)] = torch.LongTensor(comp_ingr)
            scores[i, :len(score_list)]=torch.FloatTensor(score_list)            

        # (N, top_k, ingr_embed)
        comp_emb = encoder_embedding(comp_ingr_id)

        hidden_repeat = hidden[0].unsqueeze(1).expand(-1,self.pairings.top_k,-1)
        query = self.query_layer(hidden_repeat)
        key = self.key_layer(comp_emb)

        scores_att = self.attn(torch.tanh(query + key))
        attn_weights = F.softmax(scores_att.squeeze(2), dim=-1)
        
        # TODO: try with still doing the attention, but without the scores if there's no compatible ingr ?
        # or too broad to add ingr afterwards ?
        if scores.sum() == 0:
            return None, attn_weights,comp_ingr_id
        
        # TODO: try with emphazing unknown pairings
        attn_scores = F.normalize((attn_weights * scores),1)#.to(comp_emb.device)

        # attn_scores view: (batch,1,top_k)
        # comb_emb (batch,top_k,hidden_size)
        attn_applied = torch.bmm(attn_scores.view(-1, 1, self.pairings.top_k),
                                 comp_emb)

        output = torch.cat((embedded[0], attn_applied[:, 0]), 1).unsqueeze(0)

        return output, attn_scores, comp_ingr_id
