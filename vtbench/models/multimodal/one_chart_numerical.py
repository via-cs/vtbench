import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoBranchModel(nn.Module):
    def __init__(self, chart_branch, numerical_branch, fusion_module, num_classes=2):
        super(TwoBranchModel, self).__init__()
        self.chart_branch = chart_branch
        self.numerical_branch = numerical_branch
        self.fusion = fusion_module
        self.classifier = nn.Linear(fusion_module.output_size, num_classes)  # binary classification

    def forward(self, inputs):
        chart_input, numerical_input = inputs

        chart_features = self.chart_branch(chart_input)
        numerical_features = self.numerical_branch(numerical_input)

        fused = self.fusion([chart_features, numerical_features])
        return self.classifier(fused)
