from torch.autograd import Variable
import torch.nn as nn
import torch

class CharRNN(nn.Module):
    def __init__(self,voc_size=27,emb_size=10,n_layers=1,layer_size=128, dropout=0.):
        super(CharRNN, self).__init__()
        self.n_layers = n_layers
        self.n_hid = layer_size
        self.emb_size = emb_size
        #different from karpathy - learn embedding rather than 1-hot characters
        self.emb = nn.Embedding(voc_size,emb_size)
        self.rnn = nn.LSTM(input_size=emb_size,
                           hidden_size=layer_size,
                           num_layers = n_layers,
                           dropout=dropout)
        self.lin = nn.Linear(layer_size, voc_size)

    def forward(self, inp, hidden):
        emb = self.emb(inp.view(1,-1))
        #resize input from batch x sequence to batch x sequence x embedding_size
        act, hidden = self.rnn(emb.view(-1,1,self.emb_size), hidden)
        #resize from batch x sequence x hidden to batch*sequence,hidden
        output = self.lin(act.view(-1,self.n_hid))
        #output = self.softmax(logits)
        return output, hidden

    def init_hidden(self, batch_size=1):
        h0 = Variable(torch.zeros(self.n_layers,1, self.n_hid))
        c0 = Variable(torch.zeros(self.n_layers,1, self.n_hid))
        return h0,c0
