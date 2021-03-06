import csv
import data
import shutil
import glob
import numpy as np
import os
import torch
from PIL import Image, ImageOps

def conv_output(l,size):
    w,h = size
    F = l.kernel_size
    S = l.stride
    P = l.padding
    w2= (w-F[0]+2*P[0])/S[0]+1
    h2 =(h-F[1]+2*P[1])/S[1]+1
    return w2,h2

def pool_output(l,size):
    w,h = size
    F = l.kernel_size
    S = l.stride
    P = l.padding
    w2 = (w-F)/S+1
    h2 = (h-F)/S+1
    return w2,h2

class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomRotation(object):
    """Rotate PIL.Image randomly (90/180/270 degrees)with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be rotated.
        Returns:
            PIL.Image: Randomly rotated image.
        """
        if np.random.random() < 0.5:
            deg = np.random.randint(1,3)*90.
            return img.rotate(deg)
        return img

class RandomTranslation(object):
    """Translates PIL.Image randomly (0-10 pixels) with a probability of 0.5."""

    def __init__(self,max_vshift=10, max_hshift=10):
        self.max_vshift = max_vshift
        self.max_hshift = max_hshift

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be translated.
        Returns:
            PIL.Image: Randomly translated image.
        """
        if np.random.random() < 0.5:
            hshift = np.random.randint(-self.max_hshift,self.max_hshift)
            vshift = np.random.randint(-self.max_vshift,self.max_vshift)
            return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift))
        return img

def calculate_feature_size(model,input_size):
    size = input_size
    for l in model:
        if type(l) == torch.nn.Conv2d:
            size = conv_output(l,size)
        elif type(l) == torch.nn.MaxPool2d:
            size = pool_output(l,size)
        elif type(l) == torch.nn.Sequential:
            size = calculate_feature_size(l,size)
    return size

def get_unique_labels(fname):
    labels = set()
    with open(fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(reader):
            if i == 0: # skip header
                continue
            file_id = row[0]
            file_labels = row[1].split(' ')
            labels.update(file_labels)
    return list(labels)

def count_positive_fraction(loader):
    total = 0
    counts = torch.zeros(17)
    for _,y in loader:
        counts += torch.sum(y,0)
        total += y.size()[0]
    return counts/total


def split_data(source_dir, split, dest_dirs, hard_move=True):
    '''randomly divide files in source dir over dest_dirs'''
    assert np.sum(split) == 1., 'split must sum to 1'
    assert len(split) == len(dest_dirs), 'split must be specified for each dir'
    move_fun = shutil.move if hard_move else shutil.copy
    #list all files
    files = [f for f in glob.glob(source_dir +'/*.jpg')]
    #determine number of files per destination dir
    n_files = np.floor(np.array(split)*len(files)).astype('int')
    print n_files
    #generate random order
    idx = np.random.permutation(np.arange(len(files)))
    start = 0
    #move files
    for i,dest in enumerate(dest_dirs):
        if not os.path.exists(dest):
             try:
                 os.makedirs(dest)
             except (RuntimeError, OSError):
                 print 'error creating dir'
        dest_idx = idx[start:start+n_files[i]]
        for file_idx in dest_idx:
            move_fun(files[file_idx],dest)
        start += n_files[i]
    #deal with any remainder files
    for file_idx in idx[start:]:
        move_fun(files[file_idx],np.random.choice(dest_dirs))
