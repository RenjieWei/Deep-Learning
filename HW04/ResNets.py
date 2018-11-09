import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from torch.utils import *


import os
import argparse

from resnet import myResNet


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--pre', help='load pretrained', action='store_true')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
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

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.pre:
    print('==> Loading the pretrained network')
    model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    }
    net = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,[2,2,2,2])
    net.load_state_dict(model_zoo.load_url(model_urls["resnet18"], model_dir ="./Model/"))
else:
    print('==> Loading my ResNet network')
    net = myResNet(100)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)


test_acc_ep = []
test_loss_ep = []
def train(epoch):
    print('\nTraining Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = lossFunction(outputs, targets)
        loss.backward()
        if epoch > 6:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if(state['step']>=1024):
                        state['step'] = 1000
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

def test(epoch):
    global best_acc
    print('\nTesting Epoch: %d' % epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = lossFunction(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_acc_ep.append(100.*correct/total)
        test_loss_ep.append(test_loss/(batch_idx+1))
        
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc

for epoch in range(start_epoch, start_epoch+91):
    train(epoch)
    test(epoch)
    print(test_acc_ep)
    print(test_loss_ep)
    print(best_acc)
