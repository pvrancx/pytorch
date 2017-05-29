import torch
from torch.autograd import Variable


def softmax(vals,temp):
    vals -= torch.max(vals)
    vals /= temp
    exps = torch.exp(vals)
    return exps / torch.sum(exps)

def predict(net,enc,dec,seed_text='The ',n_steps=200,temp=.5):
    enc_seed = torch.LongTensor([enc[ch] for ch in seed_text]).view(1,-1)
    h0 = net.init_hidden()
    out, h = net(Variable(enc_seed[:,-1]),h0)
    inp = Variable(enc_seed[:,:-1].view(1,-1))
    out_text = [seed_text]
    nchars = len(enc)
    for s in range(n_steps):
        out, h = net(inp,h)
        char = torch.multinomial(softmax(out[-1].data,temp),1)[0]
        out_text.append( dec[char] )
        inp= Variable(torch.LongTensor([[char]]))
    return out_text
