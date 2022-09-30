import torch
import torchvision
import numpy as np
from HodaDatasetReader import read_hoda_dataset
from data import HodaDataset

def zca(x):
    [B, C, H, W] = list(x.size())
    x = x.reshape((B, C*H*W))       # flattern the data
    mean = torch.mean(x, dim=0, keepdim=True)
    return mean

X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                             images_height=32,
                                             images_width=32,
                                             one_hot=False,
                                             reshape=False)
trainset = HodaDataset(X_train, Y_train)
trainloader = torch.utils.data.DataLoader(trainset,
                    batch_size=60000, shuffle=False, num_workers=0)

for _, data in enumerate(trainloader):
    break
images, _ = data
mean = zca(images)

torch.save(mean, './statistics/hoda_mean.pt')