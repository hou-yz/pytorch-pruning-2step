'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from heapq import nsmallest
from operator import itemgetter
import json
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from model_refactor import *
from models import *
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

use_cuda = torch.cuda.is_available()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_filter_num_pre_prune = 0

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
def train(optimizer=None, rankfilters=False):
    if optimizer is None:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        if rankfilters:
            outputs = pruner.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # try:
            #     optimizer.step()
            # except TypeError:
            #     pass

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


# test
def test(log_index=-1):
    # net.eval()
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
    print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc = 100. * correct / total

    if log_index != -1:
        delta_t, delta_t_computations, bandwidth, all_conv_computations = pruner.forward(inputs, log_index)
        data = {
            'acc': acc,
            'delta_t': delta_t,
            'delta_t_computations': int(delta_t_computations),
            'bandwidth': int(bandwidth),
            'all_conv_computations': int(all_conv_computations),
            'activation_index': log_index,
            # 'epoch': epoch if 'epoch' in globals(),
        }
        return data

    return acc


# save
def save(acc, activation_index=-1):
    print('Saving..')
    try:
        state = {
            'net': net.module if isinstance(net, torch.nn.DataParallel) else net,
            'acc': acc,
            'activation_index': activation_index,
            # 'epoch': epoch if 'epoch' in globals(),
        }
    except:
        pass
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if args.prune:
        torch.save(state, './checkpoint/ckpt.prune')
    elif args.prune_layer and activation_index != -1:
        torch.save(state, './checkpoint/ckpt.prune_layer_%d' % activation_index)
    else:
        torch.save(state, './checkpoint/ckpt.train')


class FilterPruner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}

    def forward(self, x, log_index=-1):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        delta_t_computations = 0
        all_conv_computations = 0
        t0 = time.time()
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):

                all_conv_computations += np.prod(x.data.shape[1:])
                if log_index == activation_index:
                    delta_t = time.time() - t0
                    delta_t_computations = all_conv_computations
                    bandwidth = np.prod(x.data.shape[1:])
                elif log_index == -1:
                    x.register_hook(self.compute_rank)
                    self.activations.append(x)
                    self.activation_to_layer[activation_index] = layer
                    pass
                activation_index += 1

        if log_index != -1:
            return delta_t, delta_t_computations, bandwidth, all_conv_computations
        return self.model.classifier(x.view(x.size(0), -1))

    def get_activation_index_max(self):
        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                activation_index += 1
        return activation_index

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
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def lowest_ranking_filters(self, num, activation_index):
        data = []
        if activation_index == -1:
            for i in sorted(self.filter_ranks.keys()):
                for j in range(self.filter_ranks[i].size(0)):
                    data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        else:
            for j in range(self.filter_ranks[activation_index].size(0)):
                data.append((self.activation_to_layer[activation_index], j, self.filter_ranks[activation_index][j]))
        return nsmallest(num, data, itemgetter(2))  # find the minimum of data[_][2], aka, self.filter_ranks[i][j]

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_pruning_plan(self, num_filters_to_prune, activation_index):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune, activation_index)

        # After each of the k filters are pruned,
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

    def get_candidates_to_prune(self, num_filters_to_prune, activation_index):
        self.reset()
        train(rankfilters=True)
        self.normalize_ranks_per_layer()

        return self.get_pruning_plan(num_filters_to_prune, activation_index)

    def total_num_filters(self, activation_index):
        filters = 0
        i = 0
        for name, module in list(self.model.features._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                if activation_index == -1:
                    filters = filters + module.out_channels
                elif activation_index == i:
                    filters = filters + module.out_channels
                i = i + 1

        return filters

    def prune(self, activation_index=-1):
        # Get the accuracy before pruning
        acc_pre_prune = test()
        acc = acc_pre_prune

        # train(rankfilters=True)

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = pruner.total_num_filters(activation_index)

        if activation_index == -1:
            num_filters_to_prune_per_iteration = 512
        else:
            num_filters_to_prune_per_iteration = int(number_of_filters / 16)
        # iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        #
        # iterations = int(iterations * 2.0 / 3)
        #
        # print("Number of pruning iterations to reduce 67% filters", iterations)

        # for _ in range(iterations):
        while acc > acc_pre_prune * 0.95 and pruner.total_num_filters(activation_index) / number_of_filters > 0.2:
            # print("Ranking filters.. ")

            prune_targets = pruner.get_candidates_to_prune(num_filters_to_prune_per_iteration, activation_index)
            num_layers_pruned = {}  # filters to be pruned in each layer
            for layer_index, filter_index in prune_targets:
                if layer_index not in num_layers_pruned:
                    num_layers_pruned[layer_index] = 0
                num_layers_pruned[layer_index] = num_layers_pruned[layer_index] + 1

            print("Layers that will be pruned", num_layers_pruned)
            print("..............Pruning filters............. ")
            if use_cuda:
                model = self.model.cpu()
            else:
                model = self.model

            for layer_index, filter_index in prune_targets:
                model = prune_conv_layer(model, layer_index, filter_index)

            self.model = model
            if use_cuda:
                self.model = model.cuda()
                # self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
                cudnn.benchmark = True

            optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

            print("%d / %d Filters remain." % (pruner.total_num_filters(activation_index), number_of_filters))
            # test()
            print("Fine tuning to recover from pruning iteration.")
            for epoch in range(1):
                train(optimizer)
            acc = test()
            pass
            if acc <= acc_pre_prune * 0.95:
                pass

        print("Finished. Going to fine tune the model a bit more")
        for epoch in range(5):
            train(optimizer)
        test()
        pass


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--prune_layer", dest="prune_layer", action="store_true")
    parser.add_argument("--test_pruned", dest="test_pruned", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # Model
    if args.train:
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
    else:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.train')
        net = checkpoint['net']
        acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']

    if use_cuda:
        net.cuda()
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
    total_filter_num_pre_prune = pruner.total_num_filters(activation_index=-1)

    if args.prune:
        pruner.prune(activation_index=-1)
    elif args.prune_layer:
        activation_index_max = pruner.get_activation_index_max()
        for activation_index in range(activation_index_max):
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.train')
            net = checkpoint['net']
            acc = checkpoint['acc']
            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(activation_index=-1)

            # prune given layer
            pruner.prune(activation_index)
            # prune the whole model
            pruner.prune()
            acc = test()
            save(acc, activation_index)
            pass
    elif args.train or args.resume:
        for epoch in range(100):  # start_epoch, start_epoch + 20):
            print('\nEpoch: %d' % epoch)
            train()
            acc = test()
        save(acc)
    elif args.test_pruned:
        activation_index_max = pruner.get_activation_index_max()
        original_data = []
        pruned_data = []
        for activation_index in range(activation_index_max):
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.train')
            net = checkpoint['net']
            acc = checkpoint['acc']
            # net = net.module if isinstance(net, torch.nn.DataParallel) else net
            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(activation_index=-1)

            data = test(activation_index)
            original_data.append(data)

            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.prune_layer_' + str(activation_index))
            net = checkpoint['net']
            acc = checkpoint['acc']
            # net = net.module if isinstance(net, torch.nn.DataParallel) else net
            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(activation_index=-1)

            data = test(activation_index)
            pruned_data.append(data)

        with open('./checkpoint/log_original.json', 'w') as fp:
            json.dump(original_data, fp, indent=2)
        with open('./checkpoint/log_pruned.json', 'w') as fp:
            json.dump(pruned_data, fp, indent=2)


