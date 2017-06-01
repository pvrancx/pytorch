import argparse
import torch
import model
import data
import time
import shutil

model_names = sorted(name for name in model.__dict__
    if name.startswith("Planet")
    and callable(model.__dict__[name]))

print model_names

def train(net,loader,criterion,optimizer):
    net.train()
    avg_loss = 0.
    start = time.time()
    for i, (X, y) in enumerate(loader):
        input_var = torch.autograd.Variable(X)
        target_var = torch.autograd.Variable(y)

        output = net(input_var)
        loss = criterion(output, target_var)
        avg_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%50 == 0:
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
        input_var = torch.autograd.Variable(X)
        target_var = torch.autograd.Variable(y)
        output = net(input_var)
        loss = criterion(output, target_var)

        avg_loss += loss.data[0]
    return avg_loss/len(loader)

def save_model(model_state,filename='checkpoint.pth.tar',is_best=False):
    torch.save(model_state, model_state['arch']+'-'+filename)
    if is_best:
        shutil.copyfile(filename, model_state['arch']+'-best.pth.tar')

def main(args):
    # create model and optimizer
    net = model.__dict__[args.model](17)
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.MultiLabelSoftMarginLoss()

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
    train_data = data.create_dataset(args.datapath +'/train',
                                     args.datapath + '/img_labels.csv',
                                     args.datapath + '/labels.txt',
                                     crop=224)
    val_data = data.create_dataset(args.datapath +'/val',
                                   args.datapath + '/img_labels.csv',
                                   args.datapath + '/labels.txt',
                                   crop=224)

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
        train_loss = train(net,train_loader, criterion, optimizer)
        # validate
        val_loss = validate(net, val_loader, criterion)
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
    parser.add_argument("-batch_size", type=int, default=128, help="batch size")
    parser.add_argument("-resume", type=str, default=None, help="resume training model file")
    parser.add_argument("-nepochs", type=int, default=100, help="max epochs")
    parser.add_argument("-workers", type=int, default=2, help="number of data loaders")
    parser.add_argument("datapath", type=str, help="data path")

    args = parser.parse_args()

    main(args)
