from torch.autograd import Variable
import torch.nn as nn
import torch
import util
import copy
from torchvision.models import resnet


def make_conv(nin,nout,kernel_size=3,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(nout),
        nn.PReLU(),
    )



class PlanetNet(nn.Module):
    def __init__(self,input_size=(224,224),num_labels=17, dropout=0.4, feature_maps=256):
        super(PlanetNet, self).__init__()
        self.cfg = copy.copy(locals())
        self.cfg.pop('self')
        self.dropout = dropout
        self.feature_maps =feature_maps
        self.input_size = input_size
        self.n_labels = num_labels
        self.features = self._build_features()


        self.feature_size = util.calculate_feature_size(self.features,self.input_size)
        self.classifier = self._build_classifier()

    def _build_features(self):
        return nn.Sequential(
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
    def _build_classifier(self):
        return nn.Sequential(
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
        print 'Light'

        super(PlanetNetLight, self).__init__(**kwargs)
    def _build_features(self):
        return nn.Sequential(
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

    def _build_classifier(self):
        return nn.Sequential(
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

class PlanetNetNiN(PlanetNet):
    def __init__(self,**kwargs):
        super(PlanetNetNiN, self).__init__(**kwargs)

    def _build_features(self):
        return nn.Sequential(
            make_conv(3,8,kernel_size=1,stride=1,padding=0),
            make_conv(8,32,kernel_size=5,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv(32,8,kernel_size=1,stride=1,padding=0),
            make_conv(8,8,kernel_size=5,stride=1,padding=0),
            make_conv(8,64,kernel_size=1,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv(64,64,kernel_size=1,stride=1,padding=0),
            make_conv(64,self.feature_maps,kernel_size=3,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _build_classifier(self):
        return  nn.Sequential(
            nn.Linear(self.feature_maps*self.feature_size[0] * self.feature_size[1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, self.n_labels),
        )
class PlanetNetSmall(PlanetNet):
    def __init__(self,**kwargs):
        super(PlanetNetSmall, self).__init__(**kwargs)

    def _build_features(self):
        return nn.Sequential(
            make_conv(3,8,kernel_size=1,stride=1,padding=0),
            make_conv(8,8,kernel_size=3,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv(8,32,kernel_size=1,stride=1,padding=0),
            make_conv(32,32,kernel_size=3,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv(32,self.feature_maps,kernel_size=1,stride=1,padding=0),
            make_conv(self.feature_maps,self.feature_maps,kernel_size=5,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _build_classifier(self):
        return  nn.Sequential(
            nn.Linear(self.feature_maps*self.feature_size[0] * self.feature_size[1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, self.n_labels),
        )

class PlanetNetKirk(PlanetNet):

    def __init__(self,**kwargs):
        super(PlanetNetKirk, self).__init__(**kwargs)
        self.feature_maps=128

    def _build_features(self):
        return nn.Sequential(
            make_conv(3,8,kernel_size=1,stride=1,padding=0),
            make_conv(8,8,kernel_size=1,stride=1,padding=0),
            make_conv(8,8,kernel_size=1,stride=1,padding=0),

            make_conv(8,32,kernel_size=3,stride=1,padding=0),
            make_conv(32,32,kernel_size=1,stride=1,padding=0),
            make_conv(32,32,kernel_size=1,stride=1,padding=0),
            make_conv(32,32,kernel_size=1,stride=1,padding=0),
            make_conv(32,32,kernel_size=3,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2, stride=1),

            make_conv(32,64,kernel_size=3,stride=1,padding=0),
            make_conv(64,64,kernel_size=1,stride=1,padding=0),
            make_conv(64,64,kernel_size=1,stride=1,padding=0),
            make_conv(64,64,kernel_size=1,stride=1,padding=0),
            make_conv(64,64,kernel_size=3,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2, stride=1),

            make_conv(64,128,kernel_size=3,stride=1,padding=0),
            make_conv(128,128,kernel_size=1,stride=1,padding=0),
            make_conv(128,128,kernel_size=1,stride=1,padding=0),
            make_conv(128,128,kernel_size=1,stride=1,padding=0),
            make_conv(128,128,kernel_size=3,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

    def _build_classifier(self):
        return  nn.Sequential(
            nn.Linear(128*self.feature_size[0] * self.feature_size[1],512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, self.n_labels),
        )

class PlanetResNet(resnet.ResNet):
    def __init__(self,input_size=(224,224),num_labels=17, dropout=0.4, feature_maps=256,**kwargs):
        super(PlanetResNet, self).__init__(resnet.BasicBlock, [2, 2, 2, 2],num_classes= num_labels, **kwargs)
        self.cfg = copy.copy(locals())
        self.cfg.pop('self')
