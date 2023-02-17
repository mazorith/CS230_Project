import torch
import torch.nn as nn
import torchvision
from PIL import Image
from AutoEncoders.AutoEncoder import Encoder, Decoder

#NOTE: We dont actually need to copy layers from the resnet, but its 2am and I'm a bit lazy :D

class Resnet50_1(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

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
        super().__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        self.decoder = Decoder(input_shape=8, layer1_shape=16, layer2_shape=32, output_shape=64)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.encoder = Encoder(input_shape=64, layer1_shape=32, layer2_shape=16, output_shape=8)

        del model

    def forward(self, features, shape):
        output = self.decoder(features, shape)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.encoder(output)
        return output

class Resnet50_3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        self.decoder = Decoder(input_shape=8, layer1_shape=16, layer2_shape=32, output_shape=64)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.encoder = Encoder(input_shape=1024, layer1_shape=512, layer2_shape=128, output_shape=64)

        del model

    def forward(self, features, shape):
        output = self.decoder(features, shape)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.encoder(output)
        return output

class Resnet50_4(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        self.decoder = Decoder(input_shape=64, layer1_shape=128, layer2_shape=512, output_shape=1024)
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

        del model

    def forward(self, features, shape):
        output = self.decoder(features, shape)
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

    res1.encoder.load_state_dict(torch.load('AutoEncoders\\encoder_conv1.pt'))
    res2.decoder.load_state_dict(torch.load('AutoEncoders\\decoder_conv1.pt'))

    res2.encoder.load_state_dict(torch.load('AutoEncoders\\encoder_maxpool.pt'))
    res3.decoder.load_state_dict(torch.load('AutoEncoders\\decoder_maxpool.pt'))

    res3.encoder.load_state_dict(torch.load('AutoEncoders\\encoder_layer3.pt'))
    res4.decoder.load_state_dict(torch.load('AutoEncoders\\decoder_layer3.pt'))

    res1.encoder.requires_grad = False
    res2.decoder.requires_grad = False
    res2.encoder.requires_grad = False
    res3.decoder.requires_grad = False
    res3.encoder.requires_grad = False
    res4.decoder.requires_grad = False

    optimizer = torch.optim.Adam(list(res1.parameters()) + list(res2.parameters()) +
                                list(res3.parameters()) + list(res3.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    train_data_path ="J:\\coco2017\\train2017\\"
    train_json_path ="J:\\coco2017\\annotations\\instances_train2017.json"

    test_data_path ="J:\\coco2017\\val2017\\"
    test_json_path ="J:\\coco2017\\annotations\\instances_val2017.json"

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    #Need to switch datasets from coco to ImageNet

    coco_train_data = torchvision.datasets.CocoDetection(root = train_data_path, annFile= train_json_path, 
                                                        transform=transform)
    coco_train_dataloader = torch.utils.data.DataLoader(coco_train_data, batch_size=1, shuffle=True)

    coco_test_data = torchvision.datasets.CocoDetection(root = test_data_path, annFile= test_json_path, 
                                                        transform=transform)
    coco_test_dataloader = torch.utils.data.DataLoader(coco_test_data, batch_size=1, shuffle=True)

    epochs = 4
    for epoch in range(epochs):
        loss = 0
        i = 0
        for features, labels in coco_train_dataloader:
            i += 1 #increase by batchsize
            optimizer.zero_grad()
            
            outs, enc_shape = res1(features)
            outs, enc_shape = res2(outs, enc_shape)
            outs, enc_shape = res3(outs, enc_shape)
            outs = res4(outs, enc_shape)

            print(labels)
            print(outs.shape)
            train_loss = criterion(torch.max(outs), labels)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()

            if (i/4)%40 == 0:
                print("epoch : {}, data : {}, loss = {:.6f}".format(epoch + 1, i, (loss/i)))

        loss = loss/i
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        torch.save(res1.state_dict(), 'CustomRes50_1.pt')
        torch.save(res2.state_dict(), 'CustomRes50_2.pt')
        torch.save(res3.state_dict(), 'CustomRes50_3.pt')
        torch.save(res4.state_dict(), 'CustomRes50_4.pt')