# v1 feature extractor with 2 conv layers+
# import torch
# import torch.nn as nn

# class FeatureExtractor(nn.Module):
#     def __init__(self, input_channels):  
#         super(FeatureExtractor, self).__init__()  

#         self.features = nn.Sequential(
#             nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout2d(0.1),

#             nn.Conv2d(8, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout2d(0.1)
#         )

#         self.global_pool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.global_pool(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         return x


# class MultiBranchCNN(nn.Module):
#     def __init__(self, input_channels, num_classes):  
#         super(MultiBranchCNN, self).__init__()  

#         self.bar_branch = FeatureExtractor(input_channels)
#         self.line_branch = FeatureExtractor(input_channels)
#         self.area_branch = FeatureExtractor(input_channels)
#         self.scatter_branch = FeatureExtractor(input_channels)

#         self.feature_size = 16 

#         self.classifier = nn.Sequential(
#             nn.Linear(self.feature_size * 4, 32),  
#             nn.BatchNorm1d(32),Feature 
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, num_classes)
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):  
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)  
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
                
#     def forward(self, bar_img, line_img, area_img, scatter_img):
#         # Get features from each branch
#         bar_features = self.bar_branch(bar_img)
#         line_features = self.line_branch(line_img)
#         area_features = self.area_branch(area_img)
#         scatter_features = self.scatter_branch(scatter_img)
        
#         # Concatenate features
#         combined = torch.cat([
#             bar_features, 
#             line_features, 
#             area_features, 
#             scatter_features
#         ], dim=1)
        
#         # Classification
#         output = self.classifier(combined)
#         return output

# # v2 - same architecture used after feature fusion, same as feature extractor
# import torch
# import torch.nn as nn

# class FeatureExtractor(nn.Module):
#     def __init__(self, input_channels):  
#         super(FeatureExtractor, self).__init__()  

#         self.features = nn.Sequential(
#             nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Dropout2d(0.1),

#             nn.Conv2d(8, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Dropout2d(0.1)
#         )

#         self.global_pool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.global_pool(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         return x


# class MultiBranchCNN(nn.Module):
#     def __init__(self, input_channels, num_classes):  
#         super(MultiBranchCNN, self).__init__()  

#         # Branches
#         self.bar_branch = FeatureExtractor(input_channels)
#         self.line_branch = FeatureExtractor(input_channels)
#         self.area_branch = FeatureExtractor(input_channels)
#         self.scatter_branch = FeatureExtractor(input_channels)

#         # Feature fusion size after concatenation
#         self.feature_size = 16  # Output size of each branch
#         self.fusion_size = self.feature_size * 4  # 4 branches concatenated

#         # Post-fusion feature extractor
#         # The input_channels for this instance is `1` since it's acting on a combined vector
#         self.post_fusion_extractor = FeatureExtractor(self.fusion_size)  

#         # Final classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(self.feature_size, 32),  
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, num_classes)
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):  
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)  
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, bar_img, line_img, area_img, scatter_img):
#         # Feature extraction for each input type
#         bar_features = self.bar_branch(bar_img)
#         line_features = self.line_branch(line_img)
#         area_features = self.area_branch(area_img)
#         scatter_features = self.scatter_branch(scatter_img)

#         # Feature fusion (concatenate features)
#         combined = torch.cat([bar_features, line_features, area_features, scatter_features], dim=1)

#         # Post-fusion feature extraction
#         # Reshape to [batch_size, 1, height, width] for convolutional layers
#         combined = combined.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 64, 1, 1]
#         fused_features = self.post_fusion_extractor(combined)  # Shape: [batch_size, 16]

#         # Classification
#         output = self.classifier(fused_features)
#         return output

# V3 - SimpleCNN --> SimpleCNN
# import torch
# import torch.nn as nn

# class Simple2DCNN(nn.Module):
#     def __init__(self, input_channels, num_classes=None):
#         super(Simple2DCNN, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         self.flatten_size = self._get_flatten_size(input_channels)

#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.flatten_size, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = self.fc_layers(x)
#         return x

#     def _get_flatten_size(self, input_channels):
#         dummy_input = torch.zeros(1, input_channels, 64, 64)
#         x = self.conv_layers(dummy_input)
#         return x.view(1, -1).size(1)


# class MultiBranchCNN(nn.Module):
#     def __init__(self, input_channels, num_classes):
#         super(MultiBranchCNN, self).__init__()
        
#         # Use Simple2DCNN for each branch
#         self.bar_branch = Simple2DCNN(input_channels)
#         self.line_branch = Simple2DCNN(input_channels)
#         self.area_branch = Simple2DCNN(input_channels)
#         self.scatter_branch = Simple2DCNN(input_channels)
        
#         # Feature sizes
#         self.feature_size = 64  # Output size of each Simple2DCNN branch
#         self.fusion_size = self.feature_size * 4  # Concatenated size

#         # Final classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(self.fusion_size, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, bar_img, line_img, area_img, scatter_img):
#         # Process each input through its branch
#         bar_features = self.bar_branch(bar_img)
#         line_features = self.line_branch(line_img)
#         area_features = self.area_branch(area_img)
#         scatter_features = self.scatter_branch(scatter_img)

#         # Concatenate branch outputs
#         combined = torch.cat([bar_features, line_features, area_features, scatter_features], dim=1)

#         # Classification
#         output = self.classifier(combined)
#         return output


# V4 4DeepCNN --> DeepCNN
import torch
import torch.nn as nn

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
    def __init__(self, input_channels, num_classes):
        super(MultiBranchCNN, self).__init__()
        
        # Branches with Deep2DCNN
        self.bar_branch = Deep2DCNN(input_channels)
        self.line_branch = Deep2DCNN(input_channels)
        self.area_branch = Deep2DCNN(input_channels)
        self.scatter_branch = Deep2DCNN(input_channels)
        
        # Feature sizes
        self.feature_size = 256  # Output size of Deep2DCNN branches
        self.fusion_size = self.feature_size * 4  # Concatenated size

        # Post-fusion feature extractor
        self.post_fusion_extractor = nn.Sequential(
            nn.Conv2d(self.fusion_size, 128, kernel_size=1),  # Use 1x1 conv to reduce channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Ensure output size is [batch_size, 128, 1, 1]
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the post-fusion extractor
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, bar_img, line_img, area_img, scatter_img):
        bar_features = self.bar_branch(bar_img)
        line_features = self.line_branch(line_img)
        area_features = self.area_branch(area_img)
        scatter_features = self.scatter_branch(scatter_img)

        # Concatenate branch outputs
        combined = torch.cat([bar_features, line_features, area_features, scatter_features], dim=1)

        # Post-fusion feature extraction
        combined = combined.view(combined.size(0), self.fusion_size, 1, 1)
        fused_features = self.post_fusion_extractor(combined)

        # Classification
        output = self.classifier(fused_features)
        return output
