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


def write_prediction(dataset,predictions, thresholds,filename):
    with open(filename,'w') as f:
        f.write('image_name,tags\n')
        for i,img_id in enumerate(dataset.ids):
            all_ids = np.arange(len(dataset.labels))
            label_ids = all_ids[predictions[i,].numpy()>np.array(thresholds)]
            tags = [ dataset.idx2labels[idx] for idx in label_ids]
            f.write(img_id +','+' '.join(tags)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-threshold", type=str, default='best', help="decision threshold")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-outfile", type=str, default='pred.csv', help="output file")
    #parser.add_argument("-target_file", type=str, default=None, help="ground truth file")

    parser.add_argument("model", type=str, help="model file")
    parser.add_argument("data", type=str, help="data path")

    args = parser.parse_args()

    test_data = data.create_dataset(args.data,None,'data/labels.txt',
    False,crop=224)

    val_data = data.create_dataset('data/val','data/img_labels.csv','data/labels.txt',
    crop=224)


    restored = torch.load(args.model)
    cfg = restored['cfg']

    net = model.PlanetNet(test_data.n_labels, **cfg)
    net = model.__dict__[restored['arch']](17)
    #net.load_state_dict(restored['state_dict'])


    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False )



    if args.threshold == 'optimize':
        #load validaton set
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size, shuffle=False,
            num_workers=2)
        fname = args.model+".pkl"
        #get model predictions on val set
        if os.path.exists(fname):
            with open(fname,'rb') as f:
                valp,valy = pickle.load(f)
        else:
            valp,valy = predict(net,val_loader)
            with open(fname,'wb') as f:
                pickle.dump((valp,valy),f,-1)
        #optimize thresholds
        thresholds = optimize_thresholds(valy,valp)
        print 'best thresholds:'
        print thresholds
        print 'score %f'%score_threshold(valy,valp,thresholds)
        acc = label_accuracy(valy,valp,thresholds)
        prec,rec = label_precision_recall(valy,valp,thresholds)

        for i,l in enumerate(val_data.labels):
            print('label %s \t'
                  'acc %f \t'
                  'prec %f \t'
                  'rec %f \t'%(l,acc[i],prec[i],rec[i])
            )

    else:
        thresholds = [float(args.threshold)] *17


    p,y = predict(net,test_loader)
    write_prediction(test_data,p, thresholds, args.outfile)
