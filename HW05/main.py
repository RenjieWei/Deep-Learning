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
start_epoch = 0

train_loss_ep = []
train_acc_ep = []
test_acc_ep = []


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

lossFunction = nn.TripletMarginLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

def train(epoch):
    print('\nTraining Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (anchor, pos, neg, _, _, _) in enumerate(triplet_trainloader):
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        out_achnor = net(anchor)
        out_pos = net(pos)
        out_neg = net(neg)
        loss = lossFunction(out_achnor, out_pos, out_neg)
        loss.backward()
        if epoch > 2:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if(state['step']>=1024):
                        state['step'] = 1000
        optimizer.step()

        train_loss += loss.item()
        print(batch_idx, len(triplet_trainloader), "Loss: %.3f" % (train_loss/(batch_idx+1)))
    train_loss_ep.append(train_loss/len(triplet_trainloader))

def get_codebook(epoch):
    n_code = n_triplets_per_sample * len(triplet_trainset)
    codebook = np.zeros([n_code, 1000])
    classbook = np.zeros(n_code)
    imagebook = [""] * n_code
    net.eval()
    with torch.no_grad():
        for batch_idx, (img, _, _, pos_class, _, img_path) in enumerate(triplet_trainloader):
            img = img.to(device)
            out_img = net(img)
            codebook[batch_idx*batch_size:(batch_idx+1)*batch_size, :] = out_img
            classbook[batch_idx*batch_size:(batch_idx+1)*batch_size] = pos_class
            imagebook[batch_idx*batch_size:(batch_idx+1)*batch_size] = img_path
    np.savetxt("{}/codebook_{}.np".format(model_dir, epoch), codebook)
    np.savetxt("{}/classbook_{}.np".format(model_dir, epoch), classbook)
    with open("{}/imagebook_{}.pkl".format(model_dir, epoch), "wb") as fout:
        pkl.dump(imagebook, fout)
    return codebook, classbook, imagebook

from datetime import datetime

def get_test_acc(epoch, codebook, classbook):
    print('\nTesting Epoch: %d' % epoch)

    def get_correct(output, target):
        top_idx = np.argsort(np.sum(np.abs(codebook-output)**2, axis=1))[:TOP30]
        top_prd = classbook[top_idx]
        correct = np.sum([top_prd == target])
        return correct

    overall_correct, overall_total  = 0, 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            print("Finished inference")
            correct, total = 0, len(inputs)*TOP30
            for i in range(len(outputs)):
                correct += get_correct(outputs[i], targets[i])
            overall_correct += correct
            overall_total += total
            print(batch_idx, len(testloader), 'Acc: %.3f (%d/%d)' % (correct/total, correct, total))
    print('Overall Acc: %.3f (%d/%d)' % (overall_correct/overall_total, overall_correct, overall_total))
    test_acc_ep.append((epoch, overall_correct/overall_total))


test_epoch = [1, 10, 30, 60, 90]

for epoch in range(start_epoch, start_epoch+91):
    train(epoch)
    if epoch in test_epoch:
        torch.save(net.state_dict(), "{}/checkpoint_{}.pth".format(model_dir, epoch))
        print("==> getting code")
        codebook, classbook, imagebook = get_codebook(epoch)
        print("==> finished getting code")
        get_test_acc(epoch, codebook, classbook)
    print(train_loss_ep)
    print(test_acc_ep)
