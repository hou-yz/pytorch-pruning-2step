'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from heapq import nsmallest
from operator import itemgetter
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from model_refactor import *
from models import *
from utils import progress_bar
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
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


# Training
def train(rankfilters=False):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        if rankfilters:
            outputs = prunner.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            try:
                optimizer.step()
            except TypeError:
                pass

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        try:
            state = {
                'net': net.module if isinstance(net, torch.nn.DataParallel) else net,
                'acc': acc,
                # 'epoch': epoch if 'epoch' in globals(),
            }
        except:
            pass
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not args.prune:
            torch.save(state, './checkpoint/ckpt.train')
        else:
            torch.save(state, './checkpoint/ckpt.prune')
        best_acc = acc


class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = \
            torch.sum((activation * grad), dim=0, keepdim=True). \
                sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0,
            0].data  # compute the total 1st order taylor for each filters in a given layer

        # Normalize the rank by the filter dimensions
        values = \
            values / (activation.size(0) * activation.size(2) * activation.size(3))

        if activation_index not in self.filter_ranks:  # set self.filter_ranks[activation_index]
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_().cuda()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))  # find the minimum of data[_][2], aka, self.filter_ranks[i][j]

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.reset()
        train(rankfilters=True)
        self.normalize_ranks_per_layer()

        return self.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in list(self.model.features._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        # Get the accuracy before prunning
        test()

        # train(rankfilters=True)

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = prunner.total_num_filters()
        num_filters_to_prune_per_iteration = 256
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        # iterations = int(iterations * 2.0 / 3)

        print("Number of prunning iterations to reduce 100% filters", iterations)

        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = prunner.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            num_layers_prunned = {}  # filters to be pruned in each layer
            for layer_index, filter_index in prune_targets:
                if layer_index not in num_layers_prunned:
                    num_layers_prunned[layer_index] = 0
                num_layers_prunned[layer_index] = num_layers_prunned[layer_index] + 1

            print("Layers that will be prunned", num_layers_prunned)
            print("..............Prunning filters............. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_conv_layer(model, layer_index, filter_index)

            self.model = model.cuda()
            # net = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            # cudnn.benchmark = True

            print("%.3f%% Filters remain." % float(100 * float(prunner.total_num_filters()) / number_of_filters))
            # test()
            print("Fine tuning to recover from prunning iteration.")
            for epoch in range(1):
                train()
            test()

        print("Finished. Going to fine tune the model a bit more")
        for epoch in range(5):
            train()
        torch.save(self.model, "model_prunned")


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # Model
    if args.resume or args.prune:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.prune')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        net = VGG('VGG16')
        # net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    prunner = FilterPrunner(net.module if isinstance(net, torch.nn.DataParallel) else net)

    if args.prune:
        prunner.prune()
    else:
        for epoch in range(20):# start_epoch, start_epoch + 20):
            print('\nEpoch: %d' % epoch)
            train()
            test()
