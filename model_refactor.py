import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]


def prune_conv_layer(model, layer_index, filter_index):
    _, conv = list(model.features._modules.items())[layer_index]
    batchnorm = None
    next_conv = None
    offset = 1

    while layer_index + offset < len(list(model.features._modules.items())):  # get next conv
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            _, next_conv = res
            break
        offset = offset + 1

    res = list(model.features._modules.items())[layer_index + 1]
    if isinstance(res[1], torch.nn.modules.BatchNorm2d):
        _, batchnorm = res

    is_bias_present = False
    if conv.bias is not None:
        is_bias_present = True

    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels,
                        out_channels=conv.out_channels - 1,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=is_bias_present)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    bias_numpy = conv.bias.data.cpu().numpy()

    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_conv.bias.data = torch.from_numpy(bias).cuda()

    if next_conv is not None:
        is_bias_present = False
        if next_conv.bias is not None:
            is_bias_present = True
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                            out_channels=next_conv.out_channels,
                            kernel_size=next_conv.kernel_size,
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=is_bias_present)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()
        next_new_conv.bias.data = next_conv.bias.data

    if batchnorm is not None:
        new_batchnorm = \
            torch.nn.BatchNorm2d(conv.out_channels - 1)

        try:
            old_weights = batchnorm.weight.data.cpu().numpy()
            new_weights = new_batchnorm.weight.data.cpu().numpy()
            new_weights[:filter_index] = old_weights[:filter_index]
            new_weights[filter_index:] = old_weights[filter_index + 1:]
            new_batchnorm.weight.data = torch.from_numpy(new_weights).cuda()

            bias_numpy = batchnorm.bias.data.cpu().numpy()
            bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
            bias[:filter_index] = bias_numpy[:filter_index]
            bias[filter_index:] = bias_numpy[filter_index + 1:]
            new_batchnorm.bias.data = torch.from_numpy(bias).cuda()
        except ValueError:
            pass


    if batchnorm is not None:
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index + 1],
                         [new_batchnorm]) for i, _ in enumerate(model.features)))
        del model.features
        model.features = features


    if next_conv is not None:
        features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index, layer_index + offset],
                                 [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))

        del model.features
        del conv
        model.features = features

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index],
                             [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        one_layer_classifier = False
        for _, module in list(model.classifier._modules.items()):
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if isinstance(model.classifier, torch.nn.Linear):
            old_linear_layer = model.classifier
            one_layer_classifier = True
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        params_per_input_channel = round(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

        if one_layer_classifier:
            classifier = new_linear_layer
        else:
            classifier = torch.nn.Sequential(
                *(replace_layers(model.classifier, i, [layer_index],
                                 [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model
