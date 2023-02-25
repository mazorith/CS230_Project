import torch
import torchvision
from PIL import Image
import os
import time

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

split1 = torch.nn.Sequential(*list(model.children())[:4])
split2 = torch.nn.Sequential(*list(model.children())[4:7])
split3 = torch.nn.Sequential(*list(model.children())[7:9])
last_layer = torch.nn.Sequential(*list(model.children())[9:])

print(model)
print(split1)
print(split2)
print(split3)
print(last_layer)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((500,500)),
                                                torchvision.transforms.ToTensor()])

imagenet_data = torchvision.datasets.ImageNet('J:\\ImageNet', transform = transform)
data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=False)

file1 = open('Full_plus_Tri-Split.txt', 'w')
file2 = open('Indv_layer.txt', 'w')

print('pass')
i = 0
with torch.no_grad():
    for features, labels in data_loader:
        i += 1
        
        #full model
        start_time = time.time()
        out = model(features)
        end_time1 = time.time() - start_time

        #split1
        start_time = time.time()
        out = split1(features)
        end_time2 = time.time() - start_time

        #split2
        start_time = time.time()
        out = split2(out)
        end_time3 = time.time() - start_time

        #split3
        start_time = time.time()
        out = split3(out)
        out = torch.flatten(out, 1)
        out = last_layer(out)
        end_time4 = time.time() - start_time

        print(i, end_time1, end_time2, end_time3, end_time4)
        file1.write(str(i) + ',' + str(end_time1) + ',' + str(end_time2) + ',' + str(end_time3) + ',' + str(end_time4) + '\n')

        start_time = time.time()
        out = model.conv1(features)
        end_time1 = time.time() - start_time

        start_time = time.time()
        out = model.bn1(out)
        end_time2 = time.time() - start_time
        
        start_time = time.time()
        out = model.relu(out)
        end_time3 = time.time() - start_time
        
        start_time = time.time()
        out = model.maxpool(out)
        end_time4 = time.time() - start_time

        start_time = time.time()
        out = model.layer1(out)
        end_time5 = time.time() - start_time
        
        start_time = time.time()
        out = model.layer2(out)
        end_time6 = time.time() - start_time
        
        start_time = time.time()
        out = model.layer3(out)
        end_time7 = time.time() - start_time
        
        start_time = time.time()
        out = model.layer4(out)
        end_time8 = time.time() - start_time
        
        start_time = time.time()
        out = model.avgpool(out)
        end_time9 = time.time() - start_time
        
        start_time = time.time()
        out = torch.flatten(out, 1)
        out = model.fc(out)
        end_time10 = time.time() - start_time

        print(i, '-1', end_time1, end_time2, end_time3, end_time4, end_time5, end_time6, end_time7, end_time8, end_time9, end_time10)
        file2.write(str(i) + ',' + str(end_time1) + ',' + str(end_time2) + ',' + str(end_time3) + ',' + str(end_time4)
                    + ',' + str(end_time5) + ',' + str(end_time6) + ',' + str(end_time7) + ',' + str(end_time8) + 
                    ',' + str(end_time9) + ',' + str(end_time10) + '\n')

file1.close()
file2.close()