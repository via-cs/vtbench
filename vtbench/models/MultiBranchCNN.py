# import torch
# import torch.nn as nn

# class MultiBranchCNN(nn.Module):
#     def __init__(self, num_classes=5):
#         super(MultiBranchCNN, self).__init__()
#         self.branch_area = self._create_branch()
#         self.branch_line = self._create_branch()
#         self.branch_bar = self._create_branch()
#         self.branch_scatter = self._create_branch()

#         self.fc1 = nn.Linear(4 * 128, 64)  
#         self.fc2 = nn.Linear(64, num_classes)

#     def _create_branch(self):
#         return nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),  
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64), 
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64 * 16 * 16, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2)  
#         )
        
#     def forward(self, x_area, x_line, x_bar, x_scatter):
       
#         out_area = self.branch_area(x_area)    
#         out_line = self.branch_line(x_line)    
#         out_bar = self.branch_bar(x_bar)       
#         out_scatter = self.branch_scatter(x_scatter)  
        
#         merged = torch.cat([out_area, out_line, out_bar, out_scatter], dim=1)

#         x = torch.relu(self.fc1(merged))
#         output = self.fc2(x)

#         return output


import torch
import torch.nn as nn

class MultiBranchCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiBranchCNN, self).__init__()
        
        # Create branches with focused feature extraction
        self.branch_area = self._create_branch()
        self.branch_line = self._create_branch()
        self.branch_bar = self._create_branch()
        self.branch_scatter = self._create_branch()
        
        # Simple feature fusion (no attention)
        self.fusion = nn.Sequential(
            nn.Linear(4 * 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )
        
    def _create_branch(self):
        return nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Feature processing
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x_area, x_line, x_bar, x_scatter):
        # Process each branch
        out_area = self.branch_area(x_area)
        out_line = self.branch_line(x_line)
        out_bar = self.branch_bar(x_bar)
        out_scatter = self.branch_scatter(x_scatter)
        
        # Simple concatenation
        merged = torch.cat([out_area, out_line, out_bar, out_scatter], dim=1)
        
        # Final classification
        output = self.fusion(merged)
        return output, None  