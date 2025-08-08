from vtbench.models.chart_models.simplecnn import SimpleCNN
from vtbench.models.chart_models.deepcnn import DeepCNN

def get_chart_model(name, input_channels=3, num_classes=None):
    name = name.lower()

    if name == 'simplecnn':
        return SimpleCNN(input_channels=input_channels, num_classes=num_classes)
    elif name == 'deepcnn':
        return DeepCNN(input_channels=input_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown chart model: {name}")
