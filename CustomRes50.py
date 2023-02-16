import torch
import torch.nn as nn
import torchvision
from PIL import Image
from AutoEncoders.AutoEncoder import Encoder, Decoder

#NOTE: We dont actually need to copy layers from the resnet, but its 2am and I'm a bit lazy :D

class Resnet50_1(nn.Module):
    def __init__(self, **kwargs):
        super.__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        self.conv1 = model.conv1
        self.encoder = Encoder(input_shape=64, layer1_shape=32, layer2_shape=16, output_shape=8)

        del model

    def forward(self, features):
        output = self.conv1(features)
        output = self.encoder(output)
        return output

class Resnet50_2(nn.Module):
    def __init__(self, **kwargs):
        super.__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        self.decoder = Decoder(input_shape=8, layer1_shape=16, layer2_shape=32, output_shape=64)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.encoder = Encoder(input_shape=64, layer1_shape=32, layer2_shape=16, output_shape=8)

        del model

    def forward(self, features):
        output = self.decoder(features)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.encoder(output)
        return output

class Resnet50_3(nn.Module):
    def __init__(self, **kwargs):
        super.__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        self.decoder = Decoder(input_shape=8, layer1_shape=16, layer2_shape=32, output_shape=64)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.encoder = Encoder(input_shape=1024, layer1_shape=512, layer2_shape=128, output_shape=64)

        del model

    def forward(self, features):
        output = self.decoder(features)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.encoder(output)
        return output

class Resnet50_4(nn.Module):
    def __init__(self, **kwargs):
        super.__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        self.decoder = Decoder(input_shape=64, layer1_shape=128, layer2_shape=512, output_shape=1024)
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

        del model

    def forward(self, features):
        output = self.decoder(features)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output

if __name__ == '__main__':
    res1 = Resnet50_1()
    res2 = Resnet50_2()
    res3 = Resnet50_3()
    res4 = Resnet50_4()

    res1.encoder.requires_grad = False
    res2.decoder.requires_grad = False
    res2.encoder.requires_grad = False
    res3.decoder.requires_grad = False
    res3.encoder.requires_grad = False
    res4.decoder.requires_grad = False

    optimizer = torch.optim.Adam(list(res1.parameters()) + list(res2.parameters()) +
                                list(res3.parameters()) + list(res3.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    #TODO: Create training setupt for finetuning resnet model with autoencoders