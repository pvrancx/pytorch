import csv
import data
import shutil
import glob
import numpy as np
import os

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
