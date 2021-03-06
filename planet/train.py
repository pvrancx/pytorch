import argparse
import torch
from torch.autograd import Variable
import model
import util
import data
import time
import torchvision.transforms as transforms

import shutil

model_names = sorted(name for name in model.__dict__
    if name.startswith("Planet")
    and callable(model.__dict__[name]))

print model_names

# def weighted_binary_cross_entropy(output, target, weights=None):
#
#     if weights is not None:
#         assert len(weights) == 2
#
#         loss = weights[1] * (target * torch.log(output)) + \
#                weights[0] * ((1 - target) * torch.log(1 - output))
#     else:
#         loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
#
#     return torch.neg(torch.mean(loss))

def weighted_multi_label_loss(p,y):
    return torch.neg(torch.mean(y*torch.log(p+1e-10)*0.1
                                +(1.-y)*torch.log(1.-p+1e-10)))

# class WeightedMultiLabelLoss(torch.nn.modules.loss._WeightedLoss):
#
#     def forward(self, input, target):
#         #_assert_no_grad(target)
#         weight = Variable(torch.zeros(input.size()))#self.weight.repeat(input.size(0),1))
#         return weighted_multi_label_loss(torch.sigmoid(input), target,
#                                weight)

def train(net,loader,criterion,optimizer,decay=0.):
    net.train()
    avg_loss = 0.
    start = time.time()
    for i, (X, y) in enumerate(loader):
        input_var = torch.autograd.Variable(X)
        target_var = torch.autograd.Variable(y)
    #    weights = torch.autograd.Variable(weight.repeat(X.size(0),1),requires_grad=False)

        output = net(input_var)
        #loss = weighted_multi_label_loss(torch.sigmoid(output),target_var)
        loss = criterion(output, target_var)
        avg_loss += loss.data[0]

        l1_crit = torch.nn.L1Loss(size_average=False)
        reg_loss = 0
        for param in net.parameters():
            reg_loss += l1_crit(param,Variable(torch.zeros(param.size()),requires_grad=False))
        loss += decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%20 == 0:
            dt = time.time()-start
            pct = float(i+1)/len(loader)
            curr_loss = avg_loss / (i+1)
            print('%fs elapsed \t'
                  '%f  done \t'
                  '%f loss \t'
                  '%fs remaining'%(dt,pct*100,curr_loss,dt/pct*(1.-pct)))
    return avg_loss / len(loader)

def validate(net,loader,criterion):
    net.eval()
    avg_loss = 0.
    for i, (X, y) in enumerate(loader):
        input_var = torch.autograd.Variable(X, volatile=True) #no backprop
        target_var = torch.autograd.Variable(y)
        output = net(input_var)
        #weights = torch.autograd.Variable(weight.repeat(X.size(0),1),requires_grad=False)

        output = net(input_var)
        #loss = weighted_multi_label_loss(torch.sigmoid(output),target_var)
        loss = criterion(output, target_var)

        avg_loss += loss.data[0]
    return avg_loss/len(loader)

def save_model(model_state,filename='checkpoint.pth.tar',is_best=False):
    fname = model_state['arch']+'-'+filename
    torch.save(model_state, fname)
    if is_best:
        shutil.copyfile( fname, model_state['arch']+'-best.pth.tar')

def main(args):
    # create model and optimizer
    train_trans = []
    val_trans = []
    debug_trans=[]
    siz = (256,256)
    if args.flip:
        train_trans.append(transforms.RandomHorizontalFlip())
        train_trans.append(util.RandomVerticalFlip())
    if args.rotate:
        train_trans.append(util.RandomVerticalFlip())
    if args.translate:
        train_trans.append(util.RandomTranslation())
    if args.scale > 0:
        train_trans.append(transforms.CenterCrop(224))
        train_trans.append(transforms.Scale(args.scale))
        val_trans.append(transforms.CenterCrop(224))
        val_trans.append(transforms.Scale(args.scale))
        debug_trans.append(transforms.CenterCrop(224))
        debug_trans.append(transforms.Scale(args.scale))
        siz = (args.scale,args.scale)
    if args.crop > 0:
        train_trans.append(transforms.RandomCrop(args.crop))
        val_trans.append(transforms.CenterCrop(args.crop))
        debug_trans.append(transforms.CenterCrop(args.crop))

        siz = (args.crop,args.crop)

    train_trans.append(transforms.ToTensor())
    #train_trans.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    val_trans.append(transforms.ToTensor())
    debug_trans.append(transforms.ToTensor())
#    val_trans.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))


    net = model.__dict__[args.model](input_size=siz,num_labels=17,dropout=args.dropout,feature_maps=args.features)
    print net
    optimizer = torch.optim.Adam(net.parameters(),weight_decay=args.l2_decay)
    #stats = torch.load('positive.pth.tar')
    #weights = (1.-stats['positive'])/stats['positive']
    #criterion = WeightedMultiLabelLoss(weight = weights)
    criterion = torch.nn.MultiLabelSoftMarginLoss()#torch.nn.BCELoss()#torch.nn.MultiLabelSoftMarginLoss()
    print net.feature_size

    #optionally restore weights
    if args.resume is not None:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        best_loss = checkpoint['score']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        best_loss = 1e10

    # load data
    debug_data = data.PlanetData(args.datapath +'/debug',
                                         args.datapath + '/img_labels.csv',
                                         args.datapath + '/labels.txt',
                                         transform=debug_trans)
    train_data = data.PlanetData(args.datapath +'/train',
                                     args.datapath + '/img_labels.csv',
                                     args.datapath + '/labels.txt',
                                     transform=train_trans)
    val_data = data.PlanetData(args.datapath +'/val',
                                   args.datapath + '/img_labels.csv',
                                   args.datapath + '/labels.txt',
                                   transform=val_trans)
    debug_loader = torch.utils.data.DataLoader(
        debug_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    #run training
    patience = args.patience
    for e in range(args.nepochs):
        start = time.time()
        # run 1 training epoch
        if args.debug:
            train_loss = train(net,debug_loader, criterion, optimizer, decay=args.l1_decay)
            val_loss = 0.
        else:
            train_loss = train(net,train_loader, criterion, optimizer, decay=args.l1_decay)
            val_loss = validate(net, val_loader, criterion)

        # validate
        end = time.time()
        #checkpoint
        print ('epoch %d \t'
               'time %f \t'
               'train loss %f \t'
               'val loss %f \t'%(e,end-start,train_loss, val_loss)
        )
        model_state = {
            'epoch': e,
            'score': val_loss,
            'cfg': net.cfg,
            'arch': args.model,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_model(model_state,'checkpoint.pth.tar', val_loss < best_loss)
        #early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                print('early_stopping')
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default='PlanetNet', help="model name")
    parser.add_argument("-patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("-crop", type=int, default=0, help="crop size")
    parser.add_argument("-scale", type=int, default=0, help="scale size")
    parser.add_argument("-features", type=int, default=64, help="feature maps")
    parser.add_argument("-flip", type=bool, default=True, help="random flips")
    parser.add_argument("-rotate", type=bool, default=True, help="random rotation")
    parser.add_argument("-translate", type=bool, default=True, help="random translation")
    parser.add_argument("-debug", action="store_true", help="run on debug set")


    parser.add_argument("-dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("-l1_decay", type=float, default=0., help="l1 weight decay")
    parser.add_argument("-l2_decay", type=float, default=0., help="l2 weight decay")

    parser.add_argument("-batch_size", type=int, default=128, help="batch size")
    parser.add_argument("-resume", type=str, default=None, help="resume training model file")
    parser.add_argument("-nepochs", type=int, default=100, help="max epochs")
    parser.add_argument("-workers", type=int, default=2, help="number of data loaders")
    parser.add_argument("datapath", type=str, help="data path")

    args = parser.parse_args()

    main(args)
