import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChartModel(nn.Module):
    def __init__(self, chart_branches, fusion_module):
        super(MultiChartModel, self).__init__()
        self.chart_branches = nn.ModuleList(chart_branches)
        self.fusion = fusion_module
        self.classifier = nn.Linear(fusion_module.output_size, 2)

    def forward(self, inputs):
        # inputs: list of chart inputs
        chart_features = [branch(inp) for branch, inp in zip(self.chart_branches, inputs)]
        fused = self.fusion(chart_features)
        return self.classifier(fused)
