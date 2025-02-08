import torch
import torch.nn as nn

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


class SimpleMultiBranchCNN(nn.Module):
    def __init__(self, input_channels, num_classes, fusion_method = "concatenation"):
        super(SimpleMultiBranchCNN, self).__init__()

        self.fusion_method = fusion_method
        
        self.bar_branch = Simple2DCNN(input_channels)
        self.line_branch = Simple2DCNN(input_channels)
        self.area_branch = Simple2DCNN(input_channels)
        self.scatter_branch = Simple2DCNN(input_channels)
        
        self.feature_size = 64  
        self.fusion_size = self.feature_size * 4  
        
        if fusion_method == "weighted_sum":
            self.fusion_weights = nn.Parameter(torch.ones(4))

        classifier_input_size = self.fusion_size if fusion_method == "concatenation" else self.feature_size

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, bar_img, line_img, area_img, scatter_img):
       
        bar_features = self.bar_branch(bar_img)
        line_features = self.line_branch(line_img)
        area_features = self.area_branch(area_img)
        scatter_features = self.scatter_branch(scatter_img)
    
        if self.fusion_method == "concatenation":
            combined = torch.cat([bar_features, line_features, area_features, scatter_features], dim=1)
            
        elif self.fusion_method == "weighted_sum":
            fusion_weights = torch.softmax(self.fusion_weights, dim=0)
            fusion_weights = fusion_weights.unsqueeze(0).expand(bar_features.size(0), -1)
            combined = (
                fusion_weights[:, 0].unsqueeze(1) * bar_features +
                fusion_weights[:, 1].unsqueeze(1) * line_features +
                fusion_weights[:, 2].unsqueeze(1) * area_features +
                fusion_weights[:, 3].unsqueeze(1) * scatter_features
            )
        else:
            raise ValueError(f"Invalid fusion method: {self.fusion_method}")
        output = self.classifier(combined)
        return output



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

class DeepMultiBranchCNN(nn.Module):
    def __init__(self, input_channels, num_classes, fusion_method = "concatenation"):
        super(DeepMultiBranchCNN, self).__init__()
        
        self.fusion_method = fusion_method

        self.bar_branch = Deep2DCNN(input_channels)
        self.line_branch = Deep2DCNN(input_channels)
        self.area_branch = Deep2DCNN(input_channels)
        self.scatter_branch = Deep2DCNN(input_channels)

       
        self.feature_size = 256  
        self.fusion_size = self.feature_size * 4  
        
        if fusion_method == "weighted_sum":
            self.fusion_weights = nn.Parameter(torch.ones(4))

        classifier_input_size = self.fusion_size if fusion_method == "concatenation" else self.feature_size

       
        self.classifier = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, num_classes)
        )

    def forward(self, bar_img, line_img, area_img, scatter_img):
        bar_features = self.bar_branch(bar_img)  
        line_features = self.line_branch(line_img)
        area_features = self.area_branch(area_img)
        scatter_features = self.scatter_branch(scatter_img)

        if self.fusion_method == "concatenation":
            combined = torch.cat([bar_features, line_features, area_features, scatter_features], dim=1)
           
        elif self.fusion_method == "weighted_sum":
            fusion_weights = torch.softmax(self.fusion_weights, dim=0)
            fusion_weights = fusion_weights.unsqueeze(0).expand(bar_features.size(0), -1)

            combined = (
                fusion_weights[:, 0].unsqueeze(1) * bar_features +
                fusion_weights[:, 1].unsqueeze(1) * line_features +
                fusion_weights[:, 2].unsqueeze(1) * area_features +
                fusion_weights[:, 3].unsqueeze(1) * scatter_features
            )
        else:
            raise ValueError(f"Invalid fusion method: {self.fusion_method}")

        output = self.classifier(combined)
        return output

