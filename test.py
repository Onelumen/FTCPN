import torch
from torch.utils.data import DataLoader
from scipy.interpolate import CubicSpline
from dataloader import *

raw_data = DataGenerator('./data',minibatch_len=10,train=False,test=False,dev=False
                     ,mytest=True)

data = DataLoader(raw_data.set,batch_size=3,shuffle=True,collate_fn=raw_data.collate)
for batch in data:
    print(batch['original'])
    print(batch['masked'])
    print(batch['interpolated'])
    print(batch['indexlist'])
    print("========================================")

