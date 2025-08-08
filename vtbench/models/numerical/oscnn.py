import torch
import torch.nn as nn
import numpy as np
import math

def calculate_mask_index(kernel_length_now, largest_kernel_length):
    right_zero_mask_length = math.ceil((largest_kernel_length-1)/2) - math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_length - kernel_length_now - right_zero_mask_length
    return left_zero_mask_length, left_zero_mask_length + kernel_length_now

def create_mask(in_channels, out_channels, kernel_length_now, largest_kernel_length):
    ind_left, ind_right = calculate_mask_index(kernel_length_now, largest_kernel_length)
    mask = np.ones((out_channels, in_channels, largest_kernel_length))
    mask[:, :, 0:ind_left] = 0
    mask[:, :, ind_right:] = 0
    return mask

def create_layer_mask(layer_param_list):
    largest_kernel_length = layer_param_list[-1][-1]
    masks = []
    init_weights = []
    biases = []
    for params in layer_param_list:
        conv = nn.Conv1d(in_channels=params[0], out_channels=params[1], kernel_size=params[2])
        ind_left, ind_right = calculate_mask_index(params[2], largest_kernel_length)
        big_weight = np.zeros((params[1], params[0], largest_kernel_length))
        big_weight[:, :, ind_left:ind_right] = conv.weight.detach().numpy()
        biases.append(conv.bias.detach().numpy())
        init_weights.append(big_weight)
        masks.append(create_mask(params[0], params[1], params[2], largest_kernel_length))
    
    return (
        np.concatenate(masks, axis=0).astype(np.float32),
        np.concatenate(init_weights, axis=0).astype(np.float32),
        np.concatenate(biases, axis=0).astype(np.float32)
    )

class OSConv1DLayer(nn.Module):
    def __init__(self, layer_parameters):
        super(OSConv1DLayer, self).__init__()

        mask, init_weight, init_bias = create_layer_mask(layer_parameters)
        
        self.weight_mask = nn.Parameter(torch.from_numpy(mask), requires_grad=False)
        max_kernel_size = mask.shape[-1]
        self.padding = nn.ConstantPad1d((max_kernel_size//2, max_kernel_size//2), 0)
        self.conv = nn.Conv1d(
            in_channels=mask.shape[1],
            out_channels=mask.shape[0],
            kernel_size=max_kernel_size
        )
        self.conv.weight = nn.Parameter(torch.from_numpy(init_weight), requires_grad=True)
        self.conv.bias = nn.Parameter(torch.from_numpy(init_bias), requires_grad=True)
        self.bn = nn.BatchNorm1d(mask.shape[0])

    def forward(self, x):
        self.conv.weight.data *= self.weight_mask
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)

class NumericalOSCNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=64):
        super(NumericalOSCNN, self).__init__()

        self.layer_param_list = [
            [[input_channels, 8, 7], [input_channels, 8, 9], [input_channels, 8, 11]],
            [[24, 32, 7], [24, 32, 9], [24, 32, 11]]
        ]

        self.layers = nn.Sequential(
            OSConv1DLayer(self.layer_param_list[0]),
            OSConv1DLayer(self.layer_param_list[1])
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(96, output_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        x = self.layers(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
