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

file = open('Full_Model_Times.txt', 'w')

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
        file.write(str(i) + ',' + str(end_time1) + ',' + str(end_time2) + ',' + str(end_time3) + ',' + str(end_time4) + '\n')

file.close()