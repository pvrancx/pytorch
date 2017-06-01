
import torch.utils.data as data
from PIL import Image
import os
import os.path
import glob
import csv
import torchvision.transforms as transforms
import torch


def get_labels(fname):
    with open(fname,'r') as f:
        labels = [t.strip() for t in f.read().split(',')]
    labels2idx = {t:i for i,t in enumerate(labels)}
    idx2labels = {i:t for i,t in enumerate(labels)}
    return labels,labels2idx,idx2labels

def load_img_labels(fname):
    img_labels = {}
    with open(fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(reader):
            if i == 0: # skip header
                continue
            file_id = row[0]
            file_labels = row[1].split(' ')
            img_labels[file_id] = file_labels
    return img_labels

class PlanetData(data.Dataset):

    def __init__(self, path, target_file, label_file, transform=None):
        self.path = path
        self.ids = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(path+'/*.jpg')]
        self.labels, self.labels2idx, self.idx2labels= get_labels(label_file)
        self.n_labels = len(self.labels)
        if target_file is None: #test set
            self.targets = None
        else:
            self.targets = load_img_labels(target_file)
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        if self.targets is None:
            # loaders only work if we have nonepty y
            target = torch.zeros(1)
        else:
            img_labels = self.targets[img_id]
            target = torch.zeros(self.n_labels)
            label_idx = torch.LongTensor([self.labels2idx[tag] for tag in img_labels])
            target[label_idx]=1

        img = Image.open(os.path.join(self.path, img_id+'.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.ids)



def create_dataset(img_path, target_file, label_file,
                    h_flip = True, scale= None, crop=None ):
    trans = []
    if h_flip:
        trans.append(transforms.RandomHorizontalFlip())

    if crop is not None:
        trans.append(transforms.RandomCrop(crop))

    trans.append(transforms.ToTensor())
    #trans.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225]))
    return PlanetData(img_path, target_file, label_file, transforms.Compose(trans))
