import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils import *

import os
import argparse
import numpy as np
import pickle as pkl

from tripletImageNetDataset import TripletImageNetDataset
from imageNetDataset import ImageNetDataset


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--blue', action='store_true', help='use bluewater')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

model_dir = "model"
n_triplets_per_sample = 1
batch_size = 64
TOP30 = 30

train_root = "./tiny-imagenet-200/train"
test_root = "./tiny-imagenet-200/val"
if args.blue:
    train_root = "/projects/training/bauh/tiny-imagenet-200/train"
    test_root = "/projects/training/bauh/tiny-imagenet-200/val"

triplet_trainset = TripletImageNetDataset(root=train_root, n_triplets_per_sample=n_triplets_per_sample, transform=transform_train)
triplet_trainloader = torch.utils.data.DataLoader(triplet_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

class_to_idx = triplet_trainset.get_class_to_idx()
print(class_to_idx)

testset = ImageNetDataset(root=test_root, annotations="val_annotations.txt", class_to_idx=class_to_idx, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

print("==> Finished loading data")

net = torchvision.models.resnet101(pretrained=True)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


from datetime import datetime


def load_codebook(epoch):
    codebook = np.loadtxt("{}/codebook_{}.np".format(model_dir, epoch))
    classbook = np.loadtxt("{}/classbook_{}.np".format(model_dir, epoch))
    with open("{}/imagebook_{}.pkl".format(model_dir, epoch), "rb") as fin:
        imagebook = pkl.load(fin)
    return codebook, classbook, imagebook

def eval_top_bottom_n(idx, n, codebook, classbook, imagebook):
    def display(n_idx, dist, msg):
        n_dist = dist[n_idx]
        n_cls = classbook[n_idx]
        n_path = imagebook[n_idx]
        print("==>", msg)
        for dist, path in zip(n_dist, n_cls, n_path):
            print(dist, n_cls, path)

    net.eval()
    with torch.no_grad():
        img, label, path = testset[idx]
        img = img.to(device)
        out_img = net(img)
        out_img = out_img.cpu().numpy()
        
        dist = np.sum(np.abs(codebook-out_img)**2, axis=1)
        
        print("==> query", idx, path)
        sort_idx = np.argsort(dist)
        top_idx = sort_idx[:n]
        display(top_idx, dist, "top10")

        bottom_idx = sort_idx[-n:]
        display(bottom_idx, dist, "bottom10")

test_epoch = 95
net.load_state_dict(torch.load("{}/checkpoint_{}.pth".format(model_dir, test_epoch)))

codebook, classbook, imagebook = load_codebook(test_epoch)
val_set = [42, 108, 1240, 3602, 9747]
for val in val_set:
    eval_top_bottom_n(val, 10, codebook, classbook, imagebook)
