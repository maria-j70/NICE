import torch

from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class HodaDataset(Dataset):
    def __init__(self, X,Y):
        'Initialization'
        self.X = X
        self.Y = Y


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        return self.X[index]


# def show_images(images, nmax=64):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.imshow(make_grid((images[:nmax]), nrow=8).permute(1, 2, 0))
#     plt.show()
# def show_batch(dl, nmax=64):
#     for images in dl:
#         show_images(images, nmax)
#         break
#
#
# X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
#                                      images_height=32,
#                                      images_width=32,
#                                      one_hot=False,
#                                      reshape=False)
# trainset = HodaDataset(X_train, Y_train)
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=64, shuffle=True, num_workers=0)
# show_batch(trainloader)