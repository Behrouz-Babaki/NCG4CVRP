#!/usr/bin/env python
# coding: utf-8

import os, shutil, random
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_TRAIN = 350 
NUM_VALID = 350
NUM_TEST = 1500

if __name__ == '__main__':

    for ldir in ('train', 'valid', 'test'):
        Path(SCRIPT_DIR+'/partitioned-data/data/%s'%ldir).mkdir(parents=True, exist_ok=True)

    idir = SCRIPT_DIR+'/../generate/generated_data/'
    fnames = list(filter(lambda x: x.endswith('.pkl'), os.listdir(idir)))
    random.shuffle(fnames)

    odir = SCRIPT_DIR+'/partitioned-data/data'

    i1 = NUM_TRAIN
    i2 = i1 + NUM_VALID
    i3 = i2 + NUM_TEST
    
    for fname in fnames[:i1]:
        shutil.move(idir+fname, '%s/train/'%odir+fname)
        
    for fname in fnames[i1:i2]:
        shutil.move(idir+fname, '%s/valid/'%odir+fname)

    for fname in fnames[i2:i3]:
        shutil.move(idir+fname, '%s/test/'%odir+fname)
