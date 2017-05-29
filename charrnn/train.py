import glob
import numpy as np
import argparse
import torch
import torch.nn as nn
import model
import predict
from torch.autograd import Variable
import os


def read_files(path):
    '''Reads all text files in directory and returns them as integer array'''
    data = []
    for f in glob.glob(path+'/*.txt'):
        print('reading file:  %s'%(f))
        fdata = open(f, 'r').read()
        data += list(fdata)
    vocab = set(data)

    encoder = { ch:i for i,ch in enumerate(vocab) }
    decoder = { i:ch for i,ch in enumerate(vocab) }

    enc_data = torch.LongTensor([encoder[ch] for ch in data])
    return enc_data, encoder, decoder


def random_batch(train_data, seq_length):
    max_idx = len(train_data)-(seq_length +1)
    start_idx = np.random.randint(max_idx)
    X = train_data[start_idx:start_idx+seq_length].view(1,-1)
    y = train_data[start_idx+1:start_idx+seq_length+1]
    return Variable(X),Variable(y)



def train(net, train_data, enc, dec, loss_fun, args):
    vocab_size = torch.max(train_data)
    seq_length = args.seq_length
    avg_loss = -np.log(1.0/vocab_size)*args.seq_length #from karpathy

    optimizer = torch.optim.Adam(net.parameters())
    n_batches = len(train_data)//args.seq_length

    if os.path.isfile('charrnn-checkpoint.pth.tar'):
        print("=> loading checkpoint ")
        checkpoint = torch.load('charrnn-checkpoint.pth.tar')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    net.train()

    for epoch in range(args.nepoch):
        it = 0
        for batch in range(n_batches):
            optimizer.zero_grad()
            X,y = random_batch(train_data, args.seq_length)
            hidden = net.init_hidden()
            output, hidden = net(X,hidden)
            loss = loss_fun(output, y)
            loss.backward()

            #nn.utils.clip_grad_norm(net.parameters(), args.clip)
            optimizer.step()

            avg_loss = .999 * avg_loss + .001 *loss.data
            print 'iteration %d loss %f'%(it,avg_loss.numpy())
            if it % 100 == 0:
                text = predict.predict(net,enc,dec)
                print 'sample %d: '%(it) + ''.join(text)
                state = {
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'loss': avg_loss,
                    'optimizer' : optimizer.state_dict(),
                    }
                torch.save(state, 'charrnn-checkpoint.pth.tar')

            it+=1




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-clip", type=float, default=.5, help="gradient clip")
    parser.add_argument("-dropout", type=float, default=0.3, help="dropout")
    parser.add_argument("-seq_length", type=int, default=100, help="sequence length")
    parser.add_argument("-nhid", type=int, default=128, help="hidden units")
    parser.add_argument("-nepoch", type=int, default=10, help="epochs")
    parser.add_argument("-nlayer", type=int, default=2, help="layers")

#    parser.add_argument("-batch_size", type=int, default=16, help="batch")

    parser.add_argument("data_path", type=str, help="data dir")

    args = parser.parse_args()


    train_data,enc, dec = read_files(args.data_path)
    net = model.CharRNN(voc_size=len(enc),
            n_layers=args.nlayer,
            layer_size=args.nhid,
            dropout=args.dropout)
    loss = nn.CrossEntropyLoss()
    train(net, train_data, enc,dec, loss, args)
