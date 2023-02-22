import os
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

#Finetuning of custom models
if __name__ == '__main__':
    orignal_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    for param in orignal_model.parameters():
        param.requires_grad = False

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

    '''We will not be using the imagenet dataloader. Since we will mainly be using cpu, the dataloader
    will take too long to actually load the data in memory. The following is manual way to work around this'''

    #imagenet_data = torchvision.datasets.ImageNet('J:\\ImageNet')
    #data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=4, shuffle=True)

    train_data_path ="J:\ImageNet\ILSVRC2012_img_train"

    val_data_path ="J:\ImageNet\ILSVRC2012_img_val"
    val_xml_path ="J:\ImageNet\Anotations\ILSVRC2012_bbox_val_v3\val"

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    imagenet_train_data_dir = os.listdir(train_data_path) #list of folders in train dataset
    imagenet_train_data_dir.sort()
    
    imagenet_train_image_lists = []
    for target in imagenet_train_data_dir:
        imagenet_train_image_lists.append(os.listdir(train_data_path + '\\' + target))

    imagenet_val_data = os.listdir(val_data_path)
    imagenet_val_targets = os.listdir(val_xml_path)

    imagenet_val_data.sort()
    imagenet_val_targets.sort()

    restore_lists = (imagenet_train_data_dir, imagenet_train_image_lists, imagenet_val_data, imagenet_val_targets)

    #sort these two paths

    

    epochs = 4
    for epoch in range(epochs):
        loss = 0
        i = 0
        for features, labels in data_loader:
            i += 1 #increase by batchsize
            optimizer.zero_grad()
            
            outs, enc_shape = res1(features)
            outs, enc_shape = res2(outs, enc_shape)
            outs, enc_shape = res3(outs, enc_shape)
            outs = res4(outs, enc_shape)

            #orginal_outs = orignal_model(features)


            #print(labels)
            #print(outs.shape)
            train_loss = criterion(outs, labels)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()

            if (i)%3 == 0:
                print("epoch : {}, data : {}, loss = {:.6f}".format(epoch + 1, i, (loss/i)))

        loss = loss/i
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        torch.save(res1.state_dict(), 'CustomRes50_1.pt')
        torch.save(res2.state_dict(), 'CustomRes50_2.pt')
        torch.save(res3.state_dict(), 'CustomRes50_3.pt')
        torch.save(res4.state_dict(), 'CustomRes50_4.pt')