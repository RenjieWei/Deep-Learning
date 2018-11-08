import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms



import os
import argparse

from cnn import VGG


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--mc', action='store_true', help='monte carlo dropout')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

sampling_times = 10
if args.mc:
    print("\n==> Using MC dropout, sampling times", sampling_times, "\n")
else:
    print("\n==> Using heuristic dropout\n")


print('==> Loading the network')
net = VGG('myVGG')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay=5e-4)


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
        if args.mc:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                sum_loss = None
                sum_outputs = None
                for _ in range(sampling_times):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = lossFunction(outputs, targets)
                    sum_outputs = outputs if sum_outputs is None else sum_outputs+outputs
                    sum_loss = loss if sum_loss is None else sum_loss+loss
                test_loss += sum_loss.item() / sampling_times
                _, predicted = sum_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        else:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

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

for epoch in range(start_epoch, start_epoch+31):
    train(epoch)
    test(epoch)
    print(test_acc_ep)
    print(test_loss_ep)
    print(best_acc)
