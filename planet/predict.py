import argparse
from sklearn.metrics import fbeta_score

def score_threshold(y,p,thresholds):
    score = 0.
    for row in range(p.size()[0]):
        p_thres = p[row,:]>thresholds
        score += fbeta_score(y[row,:],p_thres, beta=2)
    return score / p.size()[0]


def predict(net,loader):
    net.eval()
    prediction = torch.FloatTensor((len(loader),len(net.n_tags))))
    ground_truth= torch.FloatTensor((len(loader),len(net.n_tags))))
    for i, (X, y) in enumerate(loader):
        input_var = torch.autograd.Variable(X)
        prediction[i,:] = model(input_var).data
        ground_truth[i,:] = y
    return prediction, ground_truth


def write_prediction(predictions,idx2tag, threshold,filename):
    with open(filename,'w') as f:
        for i,img_id in enumerate(loader.ids)
            tag_ids = torch.arange(len(idx2tags))[predictions[i,]>threshold]
            tags = [ idx2tag[idx] for idx in tag_ids]
            f.write(img_id +', '+' '.join(tags))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-threshold", type=float, default=.5, help="decision threshold")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-outfile", type=str, default='pred.csv', help="output file")
    parser.add_argument("model", type=str, help="model file")
    parser.add_argument("data", type=str, help="data path")

    args = parser.parse_args()

    restored = torch.load(args.model)
    cfg = restored['config']
    idx2tags = restored['enc']

    net = model.PlanetModel(**cfg)
    net.load_state_dict(restored['state_dict'])


    test_data = data.create_dataset(args.data,None,args.data+'/labels.txt',None, False)

    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False )

    p,y = predict(net,test_loader)
    write_prediction(p, test_loader.idx2tags, args.threshold, args.outfile)
