import torch
import torch.nn as nn
import torch.nn.functional as F

class Small_TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes):
        super(Small_TCN, self).__init__()
        Kt = 11  # Kernel size
        pt = 0.3  # Dropout probability
        Ft = num_channels  # Number of filters
        
        self.pad0 = nn.ConstantPad1d(padding=(Kt-1, 0), value=0)
        self.conv0 = nn.Conv1d(in_channels=num_inputs, out_channels=Ft, kernel_size=Kt, bias=False)
        self.act0 = nn.ReLU()
        self.batchnorm0 = nn.BatchNorm1d(num_features=Ft)

        # First block
        dilation = 1
        self.pad1 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv1 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=Ft)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=pt)
        self.pad2 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv2 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=Ft)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=pt)
        
        # Second block
        dilation = 2
        self.pad3 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv3 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features=Ft)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=pt)
        self.pad4 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv4 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(num_features=Ft)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=pt)
        
        # Third block
        dilation = 4
        self.pad5 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv5 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(num_features=Ft)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=pt)
        self.pad6 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv6 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm6 = nn.BatchNorm1d(num_features=Ft)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=pt)

        # Flatten and linear layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self._get_flatten_size(num_inputs, 139), num_classes)  # Adjust input size accordingly

    def _get_flatten_size(self, num_inputs, seq_length):
        x = torch.zeros(1, num_inputs, seq_length)
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.pad3(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)
        x = self.dropout3(x)
        x = self.pad4(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.act4(x)
        x = self.dropout4(x)
        x = self.pad5(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.act5(x)
        x = self.dropout5(x)
        x = self.pad6(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.act6(x)
        x = self.dropout6(x)
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        # Now we propagate through the network correctly
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)

        # TCN
        # First block
        res = self.pad1(x)
        res = self.conv1(res)
        res = self.batchnorm1(res)
        res = self.act1(res)
        res = self.dropout1(res)
        res = self.pad2(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.act2(res)
        res = self.dropout2(res)

        x = res

        # Second block
        res = self.pad3(x)
        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.act3(res)
        res = self.dropout3(res)
        res = self.pad4(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.act4(res)
        res = self.dropout4(res)

        x = res

        # Third block
        res = self.pad5(x)
        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.act5(res)
        res = self.dropout5(res)
        res = self.pad6(res)
        res = self.conv6(res)
        res = self.batchnorm6(res)
        res = self.act6(res)
        res = self.dropout6(res)

        x = res

        # Linear layer to classify
        x = self.flatten(x)
        o = self.fc(x)
        return F.log_softmax(o, dim=1)
