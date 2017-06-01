import argparse
from sklearn.metrics import fbeta_score
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import data
import model
import copy
import cPickle as pickle
import os
import numpy as np
from sklearn.metrics import recall_score, precision_score

def score_threshold(y,p,thresholds):
    score = 0.
    for row in range(p.size()[0]):
        p_thres = p[row,:]> torch.FloatTensor(thresholds)
        score += fbeta_score(y[row,:].numpy(),p_thres.numpy(), beta=2)
    return score / p.size()[0]

def label_accuracy(y,p,thresholds):
    scores = np.zeros(y.size()[1])
    for row in range(p.size()[0]):
        p_thres = p[row,:]> torch.FloatTensor(thresholds)
        scores += (y[row,:].numpy()==p_thres.numpy())
    return scores / p.size()[0]


def label_precision_recall(y,p,thresholds):
    recall = np.zeros(y.size()[1])
    precision = np.zeros(y.size()[1])
    for label in range(p.size()[1]):
        p_thres = p[:,label]> thresholds[label]
        recall[label]= recall_score(y[:,label].numpy(),p_thres.numpy())
        precision[label]= precision_score(y[:,label].numpy(),p_thres.numpy())

    return precision, recall

def optimize_thresholds(y,p,nsteps=30):
    thresh = [.5]*17
    best_score = 0.
    for label in range(y.size()[1]):
        for t in np.linspace(0.,1.,nsteps):
            test_thresh = copy.copy(thresh)
            test_thresh[label] = t
            score = score_threshold(y,p,test_thresh)
            if score > best_score:
                print 'new best score %f'%score
                best_score = score
                thresh = copy.copy(test_thresh)
    return thresh


def predict(net,loader):
    net.eval()
    prediction = torch.FloatTensor(0,17)
    ground_truth= torch.FloatTensor(0,17)
    for i, (X, y) in enumerate(loader):
        input_var = torch.autograd.Variable(X)
        prediction = torch.cat((prediction, F.sigmoid(net(input_var)).data), 0)
        ground_truth = torch.cat((ground_truth, y), 0)
    return prediction, ground_truth


def write_prediction(predictions,idx2tag, threshold,filename):
    with open(filename,'w') as f:
        for i,img_id in enumerate(loader.ids):
            tag_ids = torch.arange(len(idx2tags))[predictions[i,]>threshold]
            tags = [ idx2tag[idx] for idx in tag_ids]
            f.write(img_id +', '+' '.join(tags))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-threshold", type=str, default='best', help="decision threshold")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-outfile", type=str, default=None, help="output file")
    parser.add_argument("-target_file", type=str, default=None, help="ground truth file")

    parser.add_argument("model", type=str, help="model file")
    parser.add_argument("data", type=str, help="data path")

    args = parser.parse_args()

    test_data = data.create_dataset(args.data,args.target_file,'data/labels.txt',
    False,None,crop=224)


    restored = torch.load(args.model)
    cfg = restored['cfg']

    net = model.PlanetNet(test_data.n_labels, **cfg)
    net.load_state_dict(restored['state_dict'])


    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False )

    fname = args.model+".pkl"
    if os.path.exists(fname):
        with open(fname,'rb') as f:
            p,y = pickle.load(f)
    else:
        p,y = predict(net,test_loader)
        with open(fname,'wb') as f:
            pickle.dump((p,y),f,-1)

    if args.threshold == 'best':
        thresholds = [0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.34482758620689657, 0.13793103448275862, 0.13793103448275862, 0.068965517241379309, 0.13793103448275862, 0.20689655172413793, 0.17241379310344829, 0.13793103448275862, 0.13793103448275862, 0.13793103448275862, 0.24137931034482757, 0.13793103448275862, 0.068965517241379309]
    elif args.threshold == 'optimize':
        if len(y) != 0:
            thresholds = [0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.34482758620689657, 0.13793103448275862, 0.13793103448275862, 0.068965517241379309, 0.13793103448275862, 0.20689655172413793, 0.17241379310344829, 0.13793103448275862, 0.13793103448275862, 0.13793103448275862, 0.24137931034482757, 0.13793103448275862, 0.068965517241379309]
            #thresholds = optimize_thresholds(y,p)
            print 'best thresholds:'
            print thresholds
            print 'score %f'%score_threshold(y,p,thresholds)
            acc = label_accuracy(y,p,thresholds)
            prec,rec = label_precision_recall(y,p,thresholds)

            for i,l in enumerate(test_data.labels):
                print('label %s \t'
                      'acc %f \t'
                      'prec %f \t'
                      'rec %f \t'%(l,acc[i],prec[i],rec[i])
                )
        else:
            raise RuntimeError('cannot optimize thresholds without labels')
    else:
        thresholds = [float(args.threshold)] *17



    if args.outfile is not None:
        write_prediction(p, test_data.idx2tags, thresholds, args.outfile)
