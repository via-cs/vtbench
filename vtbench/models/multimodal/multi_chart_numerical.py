import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChartNumericalModel(nn.Module):
    def __init__(self, chart_branches, numerical_branch, fusion_module):
        super(MultiChartNumericalModel, self).__init__()
        self.chart_branches = nn.ModuleList(chart_branches)
        self.numerical_branch = numerical_branch
        self.fusion = fusion_module
        self.classifier = nn.Linear(fusion_module.output_size, 2)

    def forward(self, inputs):
        # inputs: ([chart1, chart2, ...], numerical_input)
        chart_inputs, numerical_input = inputs
        chart_features = [branch(img) for branch, img in zip(self.chart_branches, chart_inputs)]
        numerical_features = self.numerical_branch(numerical_input)
        fused = self.fusion(chart_features + [numerical_features])
        return self.classifier(fused)
