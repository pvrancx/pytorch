import torch
from torch.autograd import Variable
import argparse
import model


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
    return ''.join(out_text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=str, default='The ', help="model seed text")
    parser.add_argument("-steps", type=int, default=200, help="steps")
    parser.add_argument("-temp", type=float, default=.5, help="temperature")

    parser.add_argument("model", type=str, help="model file")

    args = parser.parse_args()

    restored = torch.load(args.model)
    cfg = restored['config']
    enc = restored['enc']
    dec = restored['dec']

    net = model.CharRNN(voc_size=len(enc),
            n_layers=cfg.nlayer,
            layer_size=cfg.nhid,
            dropout=cfg.dropout)
    net.load_state_dict(restored['state_dict'])

    print predict(net,enc,dec,seed_text=args.seed,n_steps=args.steps,temp=args.temp)
