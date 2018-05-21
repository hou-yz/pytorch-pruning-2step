'''
Train & Pruning with PyTorch by hou-yz.
forked from kuangliu/pytorch-cifar;


'''

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
from heapq import nsmallest
from operator import itemgetter
import json
import numpy as np
import argparse
from models import *
from model_refactor import *

if os.name == 'nt':  # windows
    num_workers = 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:  # linux
    num_workers = 8
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

use_cuda = torch.cuda.is_available()
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
total_filter_num_pre_prune = 0
batch_size = 128

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Training
def train(optimizer=None, rankfilters=False):
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
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

        train_loss += loss.data[0]  # item()
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
    if log_index == -1 or use_cuda:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]  # loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        acc = 100. * correct / total

    if log_index != -1:
        (inputs, targets) = list(testloader)[0]
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # get profile
        with torch.autograd.profiler.profile() as prof:
            # delta_t, delta_t_computations, bandwidth, all_conv_computations = pruner.forward_n_track(Variable(inputs), log_index)
            net(Variable(inputs))
            # print(next(net.parameters()).is_cuda)
        delta_t, delta_t_computations, bandwidth0, all_conv_computations = pruner.forward_n_track(Variable(inputs),
                                                                                                  log_index)
        cfg = pruner.get_cfg()

        # get log for time/bandwidth
        delta_ts = []
        bandwidths = []
        for i in range(len(cfg)):
            delta_ts.append(
                sum(item.cpu_time for item in prof.function_events[:pruner.conv_n_pool_to_layer[i]]) / np.power(10,
                                                                                                                9) / batch_size)
            if isinstance(cfg[i], int):
                bandwidths.append(
                    int(cfg[i] * (inputs.shape[2] * inputs.shape[3]) / np.power(4, cfg[:i + 1].count('M'))))
            else:
                bandwidths.append(
                    int(cfg[i - 1] * (inputs.shape[2] * inputs.shape[3]) / np.power(4, cfg[:i + 1].count('M'))))

        data = {
            'acc': acc if use_cuda else -1,
            'index': log_index,
            # 'delta_t': delta_t,
            'delta_t_prof': delta_ts[log_index],
            'delta_ts': delta_ts,
            # 'delta_t_computations': int(delta_t_computations),
            'bandwidth': bandwidths[log_index],
            'bandwidths': bandwidths,
            # 'all_conv_computations': int(all_conv_computations),
            'layer_cfg': cfg[log_index],
            'config': cfg
            # 'epoch': epoch if 'epoch' in globals(),
        }
        return data

    return acc


# save
def save(acc, conv_index=-1, epoch=-1):
    print('Saving..')
    try:
        # save the cpu model
        model = net.module if isinstance(net, torch.nn.DataParallel) else net
        state = {
            'net': model.cpu() if use_cuda else model,
            'acc': acc,
            'conv_index': conv_index,
            'epoch': epoch,
        }
    except:
        pass
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if args.prune:
        torch.save(state, './checkpoint/ckpt.prune')
    elif args.prune_layer and conv_index != -1:
        torch.save(state, './checkpoint/ckpt.prune_layer_%d' % conv_index)
    elif epoch != -1:
        torch.save(state, './checkpoint/ckpt.train.epoch_' + str(epoch))
    else:
        torch.save(state, './checkpoint/ckpt.train')

    # restore the cuda or cpu model
    if use_cuda:
        net.cuda()


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

    # forward method that gives "compute_rank" a hook
    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        conv_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[conv_index] = layer
                conv_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    # forward method that tracks computation info
    def forward_n_track(self, x, log_index=-1):
        self.conv_n_pool_to_layer = {}

        index = 0
        delta_t_computations = 0
        all_conv_computations = 0  # num of conv computations to the given layer
        t0 = time.time()
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.ReLU) or isinstance(module, torch.nn.modules.MaxPool2d):
                all_conv_computations += np.prod(x.data.shape[1:])
                self.conv_n_pool_to_layer[index] = layer
                if log_index == index:
                    delta_t = time.time() - t0
                    delta_t_computations = all_conv_computations
                    bandwidth = np.prod(x.data.shape[1:])
                index += 1

        return delta_t, delta_t_computations, bandwidth, all_conv_computations

    # for all the conv layers
    def get_conv_index_max(self):
        conv_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            if isinstance(module, torch.nn.modules.Conv2d):
                conv_index += 1
        return conv_index

    # for all the relu layers and pool2d layers
    def get_cfg(self):
        cfg = []
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            if isinstance(module, torch.nn.modules.Conv2d):
                cfg.append(module.out_channels)
            elif isinstance(module, torch.nn.modules.MaxPool2d):
                cfg.append('M')
        return cfg

    def compute_rank(self, grad):
        conv_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[conv_index]
        values = torch.sum((activation * grad), dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[
                 0, :, 0, 0].data  # compute the total 1st order taylor for each filters in a given layer

        # Normalize the rank by the filter dimensions
        values = values / (activation.size(0) * activation.size(2) * activation.size(3))

        if conv_index not in self.filter_ranks:  # set self.filter_ranks[conv_index]
            self.filter_ranks[conv_index] = torch.FloatTensor(activation.size(1)).zero_()
            if use_cuda:
                self.filter_ranks[conv_index] = self.filter_ranks[conv_index].cuda()

        self.filter_ranks[conv_index] += values
        self.grad_index += 1

    def lowest_ranking_filters(self, num, conv_index):
        data = []
        if conv_index == -1:
            for i in sorted(self.filter_ranks.keys()):
                for j in range(self.filter_ranks[i].size(0)):
                    data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        else:
            for j in range(self.filter_ranks[conv_index].size(0)):
                data.append((self.activation_to_layer[conv_index], j, self.filter_ranks[conv_index][j]))
        return nsmallest(num, data, itemgetter(2))  # find the minimum of data[_][2], aka, self.filter_ranks[i][j]

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_pruning_plan(self, num_filters_to_prune, conv_index):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune, conv_index)

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

    def get_candidates_to_prune(self, num_filters_to_prune, conv_index):
        self.reset()
        train(rankfilters=True)
        self.normalize_ranks_per_layer()

        return self.get_pruning_plan(num_filters_to_prune, conv_index)

    def total_num_filters(self, conv_index):
        filters = 0
        i = 0
        for name, module in list(self.model.features._modules.items()):
            if isinstance(module, torch.nn.modules.Conv2d):
                if conv_index == -1:
                    filters = filters + module.out_channels
                elif conv_index == i:
                    filters = filters + module.out_channels
                i = i + 1

        return filters

    def prune(self, conv_index=-1):
        # Get the accuracy before pruning
        acc_pre_prune = test()
        acc = acc_pre_prune

        # train(rankfilters=True)

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = pruner.total_num_filters(conv_index)

        num_filters_to_prune_per_iteration = max(int(number_of_filters / 16), 2)
        # iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        #
        # iterations = int(iterations * 2.0 / 3)
        #
        # print("Number of pruning iterations to reduce 67% filters", iterations)

        # for _ in range(iterations):
        while acc > acc_pre_prune * 0.95 and pruner.total_num_filters(conv_index) / number_of_filters > 0.2:
            # print("Ranking filters.. ")

            prune_targets = pruner.get_candidates_to_prune(num_filters_to_prune_per_iteration, conv_index)
            num_layers_pruned = {}  # filters to be pruned in each layer
            for layer_index, filter_index in prune_targets:
                if layer_index not in num_layers_pruned:
                    num_layers_pruned[layer_index] = 0
                num_layers_pruned[layer_index] = num_layers_pruned[layer_index] + 1

            print("Layers that will be pruned", num_layers_pruned)
            print("..............Pruning filters............. ")
            if use_cuda:
                self.model.cpu()
            # else:
            #     model = self.model

            for layer_index, filter_index in prune_targets:
                prune_conv_layer(self.model, layer_index, filter_index)

            if use_cuda:
                self.model.cuda()
                # self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True
            # else:
            #     self.model = model

            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

            print("%d / %d Filters remain." % (pruner.total_num_filters(conv_index), number_of_filters))
            # test()
            print("Fine tuning to recover from pruning iteration.")
            for epoch in range(2):
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
    parser.add_argument('--epoch', default=10, type=int, help='epoch')
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
        start_epoch = checkpoint['epoch'] + 1

    if use_cuda:
        net.cuda()
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        # cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
    total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)

    if args.prune:
        pruner.prune()
        acc = test()
        save(acc)
        pass
    elif args.prune_layer:
        conv_index_max = pruner.get_conv_index_max()
        for conv_index in range(conv_index_max):
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.prune')
            net = checkpoint['net']
            acc = checkpoint['acc']
            if use_cuda:
                net.cuda()
                # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True

            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)

            # prune given layer
            pruner.prune(conv_index)
            # prune the whole model
            # pruner.prune()
            acc = test()
            save(acc, conv_index)
            pass
    elif args.train or args.resume:
        for epoch in range(start_epoch, start_epoch + args.epoch):
            print('\nEpoch: %d' % epoch)
            train()
            acc = test()
            if epoch % 10 == 0:
                save(acc, -1, epoch)
                pass
        save(acc)
    elif args.test_pruned:
        # use_cuda = 0
        cfg = pruner.get_cfg()
        conv_index_max = pruner.get_conv_index_max()
        original_data = []
        pruned_data = []

        last_conv_index = 0  # log for checkpoint restoring, nearest conv layer
        for index in range(len(cfg)):
            # original
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.train')
            net = checkpoint['net']
            acc = checkpoint['acc']
            if use_cuda:
                net.cuda()
                # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True

            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)

            data = test(index)
            if data['acc'] == -1:
                data['acc'] = acc
            original_data.append(data)

            # pruned
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.prune_layer_' + str(last_conv_index))
            # checkpoint = torch.load('./checkpoint/ckpt.prune')
            net = checkpoint['net']
            acc = checkpoint['acc']
            if use_cuda:
                net.cuda()
                # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True

            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)

            data = test(index)
            pruned_data.append(data)

            if index + 1 < len(cfg):
                if not isinstance(cfg[index + 1], str):
                    last_conv_index += 1

        with open('./log_original.json', 'w') as fp:
            json.dump(original_data, fp, indent=2)
        with open('./log_pruned.json', 'w') as fp:
            json.dump(pruned_data, fp, indent=2)
