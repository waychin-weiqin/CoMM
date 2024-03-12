from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
import math
from torch import nn as nn
import torch.nn.functional as F
import numpy as np

class ResNet(nn.Module):
    def __init__(self, block, layers, classes=7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.drop = nn.Dropout1d(0.5)
        # self.fc = nn.Linear(512, classes)
        self.fc = nn.Linear(2048, classes)



        # self.class_classifier = nn.utils.weight_norm(self.class_classifier, dim=0)

        # class_mask = np.random.binomial(n=1, p=0.5, size=(7, 512))
        # self.class_mask = torch.from_numpy(class_mask).float().cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def extract(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def classify(self, x, pool_flat=True):
        # x = F.normalize(x, dim=1)
        if pool_flat:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

    def cosine_forward(self, x):
        fc_weight = self.fc.weight
        fc_bias = self.fc.bias
        out = self.extract(x)
        out = self.avgpool(out)

        C = out.size(1)
        out = out.view(-1, C)
        out_norm = torch.norm(out, dim=1, keepdim=True)
        w_norm = torch.norm(fc_weight, dim=1, keepdim=True)  # [C, 1]
        outp = torch.mm(out, fc_weight.t())
        cosine = outp / (out_norm * w_norm.t())
        # outp += fc_bias

        return cosine, outp

    def norm_cosine(self, x, fc_weight=None):
        if fc_weight is None:
            fc_weight = self.fc.weight
        out = self.extract(x)
        out = self.avgpool(out)
        C = out.size(1)
        out = out.view(-1, C)
        out_norm = torch.norm(out, dim=1, keepdim=True) # [N, 1]
        w_norm = torch.norm(fc_weight, dim=1, keepdim=True)    # [C, 1]
        cosine = torch.mm(out, fc_weight.t()) / (out_norm * w_norm.t())

        return cosine, out_norm

    def forward(self, x):
        x = self.extract(x)
        x = self.classify(x)

        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.fc = nn.Linear(512, kwargs["classes"])
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        # state_dict.pop('fc.weight')
        # state_dict.pop('fc.bias')
        _ = model.load_state_dict(state_dict, strict=False)
        # _ = model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        print(_) # Missing keys
        print(f"[INFO] Loaded ImageNet Weights")
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.fc = nn.Linear(2048, kwargs["classes"])
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])

        if kwargs["classes"] != 1000:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

        _ = model.load_state_dict(state_dict, strict=False)
        print(_)
        print(f"[INFO] Loaded ImageNet Weights")
    return model
