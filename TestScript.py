import torch
from PIL import Image
import torchvision.transforms as transforms

def total_size(shape):
    size = 1
    for x in shape:
        size = size * x
    return size

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

image = Image.open('Bear1.jpg')
converter_tensor = transforms.ToTensor()
img_tensor = converter_tensor(image)
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.shape)
out = model(img_tensor)

out1 = model.conv1(img_tensor)
layer_size = total_size(out1.shape)
print(out1.shape, layer_size)

out2 = model.bn1(out1)
layer_size = total_size(out2.shape)
print(out2.shape, layer_size)

out3 = model.relu(out2)
layer_size = total_size(out3.shape)
print(out3.shape, layer_size)

out4 = model.maxpool(out3)
layer_size = total_size(out4.shape)
print(out4.shape, layer_size)

out5 = model.layer1(out4)
layer_size = total_size(out5.shape)
print(out5.shape, layer_size)

out6 = model.layer2(out5)
layer_size = total_size(out6.shape)
print(out6.shape, layer_size)

out7 = model.layer3(out6)
layer_size = total_size(out7.shape)
print(out7.shape, layer_size)

out8 = model.layer4(out7)
layer_size = total_size(out8.shape)
print(out8.shape, layer_size)

out9 = model.avgpool(out8)
layer_size = total_size(out9.shape)
print(out9.shape, layer_size)

out9 = torch.flatten(out9, 1)
out10 = model.fc(out9)
layer_size = total_size(out10.shape)
print(out10.shape, layer_size)