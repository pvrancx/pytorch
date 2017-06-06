from torch.autograd import Variable
import torch.nn as nn
import torch
import util
import copy
from torchvision.models import resnet

class PlanetNet(nn.Module):
    def __init__(self,input_size=(224,224),num_labels=17, dropout=0.4, feature_maps=256):
        super(PlanetNet, self).__init__()
        self.cfg = copy.copy(locals())
        self.cfg.pop('self')
        self.dropout = dropout
        self.feature_maps =feature_maps
        self.input_size = input_size
        self.n_labels = num_labels
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.feature_maps, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_maps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.feature_size = util.calculate_feature_size(self.features,self.input_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.feature_maps * self.feature_size[0] * self.feature_size[1], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.n_labels),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.feature_maps * self.feature_size[0] * self.feature_size[1])
        x = self.classifier(x)
        return x

class PlanetNetLight(PlanetNet):
    def __init__(self,**kwargs):
        super(PlanetNetLight, self).__init__(**kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.feature_maps, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_maps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.feature_size = util.calculate_feature_size(self.features,self.input_size)

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.feature_maps*self.feature_size[0] * self.feature_size[1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.n_labels),
        )



class PlanetNetSmall(PlanetNet):
    def __init__(self,**kwargs):
        super(PlanetNetSmall, self).__init__(**kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, self.feature_maps, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_maps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.feature_size = util.calculate_feature_size(self.features,self.input_size)

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.feature_maps*self.feature_size[0] * self.feature_size[1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.n_labels),
        )

class PlanetResNet(resnet.ResNet):
    def __init__(self,input_size=(224,224),num_labels=17, dropout=0.4, feature_maps=256,**kwargs):
        super(PlanetResNet, self).__init__(resnet.BasicBlock, [2, 2, 2, 2],num_classes= num_labels, **kwargs)
        self.cfg = copy.copy(locals())
        self.cfg.pop('self')
