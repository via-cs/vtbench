# import torch
# import torch.nn as nn

# class FeatureExtractor(nn.Module):
#     def __init__(self, input_channels):
#         super(FeatureExtractor, self).__init__()
#         self.features = nn.Sequential(
#             # First block
#             nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
            
#             # Second block
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
            
#             # Global pooling instead of third conv block
#             nn.AdaptiveAvgPool2d(1)
#         )
        
#     def forward(self, x):
#         return self.features(x)

# class MultiBranchCNN(nn.Module):
#     def __init__(self, input_channels, num_classes):
#         super(MultiBranchCNN, self).__init__()
#         self.bar_branch = FeatureExtractor(input_channels)
#         self.line_branch = FeatureExtractor(input_channels)
#         self.area_branch = FeatureExtractor(input_channels)
#         self.scatter_branch = FeatureExtractor(input_channels)
        
#         # Simpler fusion network
#         self.classifier = nn.Sequential(
#             nn.Linear(32 * 4, 64),  # Reduced dimensions
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )
        
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
    
#     def forward(self, bar_img, line_img, area_img, scatter_img):
#         # Extract features
#         bar_features = self.bar_branch(bar_img).squeeze(-1).squeeze(-1)
#         line_features = self.line_branch(line_img).squeeze(-1).squeeze(-1)
#         area_features = self.area_branch(area_img).squeeze(-1).squeeze(-1)
#         scatter_features = self.scatter_branch(scatter_img).squeeze(-1).squeeze(-1)
        
#         # Combine features
#         combined = torch.cat([
#             bar_features, line_features, 
#             area_features, scatter_features
#         ], dim=1)
        
#         return self.classifier(combined)


import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels):  
        super(FeatureExtractor, self).__init__()  

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x


class MultiBranchCNN(nn.Module):
    def __init__(self, input_channels, num_classes):  
        super(MultiBranchCNN, self).__init__()  

        self.bar_branch = FeatureExtractor(input_channels)
        self.line_branch = FeatureExtractor(input_channels)
        self.area_branch = FeatureExtractor(input_channels)
        self.scatter_branch = FeatureExtractor(input_channels)

        self.feature_size = 16 

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size * 4, 32),  
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, bar_img, line_img, area_img, scatter_img):
        # Get features from each branch
        bar_features = self.bar_branch(bar_img)
        line_features = self.line_branch(line_img)
        area_features = self.area_branch(area_img)
        scatter_features = self.scatter_branch(scatter_img)
        
        # Concatenate features
        combined = torch.cat([
            bar_features, 
            line_features, 
            area_features, 
            scatter_features
        ], dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output

