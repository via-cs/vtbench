import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# OS CNN model
def calculate_mask_index(kernel_length_now,largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_lenght)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_lenght))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_lenght)
        big_weight = np.zeros((i[1],i[0],largest_kernel_lenght))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()
        
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        
        mask = creat_mask(i[1],i[0],i[2], largest_kernel_lenght)
        mask_list.append(mask)
        
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)

    
class build_layer_with_layer_parameter(nn.Module):
    def __init__(self,layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)
        
        
        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)
        
        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)
         
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        #self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result    
    
class OS_CNN(nn.Module):
    def __init__(self,layer_parameter_list,n_class,few_shot = True):
        super(OS_CNN, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        
        
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)
        
        self.net = nn.Sequential(*self.layer_list)
            
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        
        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1] 
            
        self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X):
        
        X = self.net(X)

        X = self.averagepool(X)
        X = X.squeeze_(-1)

        if not self.few_shot:
            X = self.hidden(X)
        return X
    
# OS CNN numerical branch
class OSNumericalBranch(nn.Module):
    def __init__(self, input_channels=1, n_class=64):
        super(OSNumericalBranch, self).__init__()
        
        # Use original layer params from OS-CNN paper
        self.layer_parameter_list = [
            [[input_channels, 8, 7], [input_channels, 8, 9], [input_channels, 8, 11]],
            [[24, 32, 7], [24, 32, 9], [24, 32, 11]]
        ]
        
        # Set few_shot=False only if you want classification; we want features
        self.oscnn = OS_CNN(
            layer_parameter_list=self.layer_parameter_list,
            n_class=96,  # matches CNN feature size like 64 or 256
            few_shot=True
        )

        self.project = nn.Linear(96, n_class)

    def forward(self, x):
        # x: (B, T) or (B, 1, T)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.oscnn(x)
        return self.project(x)  # Output: (B, n_class)



class FCN(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super(FCN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.fc_layers(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, config):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Linear(input_dim, config['hidden_dim'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'], 
            nhead=config['num_heads'], 
            dim_feedforward=config['hidden_dim'] * 2, 
            dropout=config['dropout'],
            batch_first=True 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        output_size = config.get('output_size', 256) 
        self.fc_out = nn.Linear(config['hidden_dim'], output_size)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc_out(x)


class Simple2DCNN(nn.Module):
    def __init__(self, input_channels, num_classes=None):
        super(Simple2DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten_size = self._get_flatten_size(input_channels)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def _get_flatten_size(self, input_channels):
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        x = self.conv_layers(dummy_input)
        return x.view(1, -1).size(1)



class Deep2DCNN(nn.Module):
    def __init__(self, input_channels):
        super(Deep2DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x



class MultiBranchCNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_features, config):
        super(MultiBranchCNN, self).__init__()
        
        self.fusion_method = config['fusion_method']
        self.numerical_method = config['numerical_processing']['method']
        
        if config['architecture'] == "SimpleMultiBranchCNN":
            CNNModel = Simple2DCNN
            self.feature_size = 64
        else:
            CNNModel = Deep2DCNN
            self.feature_size = 256
        
        self.bar_branch = CNNModel(input_channels)
        self.line_branch = CNNModel(input_channels)
        self.area_branch = CNNModel(input_channels)
        self.scatter_branch = CNNModel(input_channels)

        if self.numerical_method == 'fcn':
            self.numerical_branch = FCN(
                input_dim=num_features,
                output_dim=self.feature_size  # Match CNN feature size
            )
            self.numerical_output_size = self.feature_size
        elif self.numerical_method == 'transformer':
            transformer_config = config['numerical_processing']['transformer']
            transformer_config['output_size'] = self.feature_size  
            
            self.numerical_branch = TransformerEncoder(
                input_dim=num_features,
                config=transformer_config
            )
            self.numerical_output_size = self.feature_size
        # numerical processing branch
        elif self.numerical_method == 'oscnn':
            self.numerical_branch = OSNumericalBranch(
                input_channels=1,
                n_class=self.feature_size
            )
            self.numerical_output_size = self.feature_size

        elif self.numerical_method in ['none', None]:
            self.numerical_branch = None
            self.numerical_output_size = 0

        else:
            raise ValueError(f"Unsupported numerical processing method: {self.numerical_method}")


        # calculating fusion size
        if self.fusion_method == 'concatenation':
            self.fusion_size = self.feature_size * 4
            self.classifier_input_size = self.fusion_size + self.numerical_output_size
        else:  # weighted_sum
            self.fusion_size = self.feature_size
            self.classifier_input_size = self.feature_size
            
        # Initialize fusion weights if using weighted sum
        if self.fusion_method == 'weighted_sum':
            num_weights = 5 if self.numerical_branch is not None else 4
            self.fusion_weights = nn.Parameter(torch.ones(num_weights))
        
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, bar_img, line_img, area_img, scatter_img, numerical_data=None):
        bar_features = self.bar_branch(bar_img)
        line_features = self.line_branch(line_img)
        area_features = self.area_branch(area_img)
        scatter_features = self.scatter_branch(scatter_img)
        
         # Process numerical data if available
        if self.numerical_branch is not None and numerical_data is not None:
            numerical_features = self.numerical_branch(numerical_data)
        else:
            numerical_features = None
        
        # Feature fusion
        if self.fusion_method == 'concatenation':
            # Concatenate CNN features
            combined = torch.cat([
                bar_features,
                line_features,
                area_features,
                scatter_features
            ], dim=1)
            
            # Add numerical features if present
            if numerical_features is not None:
                combined = torch.cat([combined, numerical_features], dim=1)
                
        else:  # weighted_sum
            # Normalize weights
            weights = F.softmax(self.fusion_weights, dim=0)
            
            # Combine CNN features with weights
            combined = (
                weights[0] * bar_features +
                weights[1] * line_features +
                weights[2] * area_features +
                weights[3] * scatter_features
            )
            
            # Add numerical features if present
            if numerical_features is not None:
                combined = combined + weights[4] * numerical_features
        
        # Classification
        output = self.classifier(combined)
        return output