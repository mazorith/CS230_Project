import torch
from PIL import Image
import torchvision.transforms as transforms

def total_size(shape):
    size = 1
    for x in shape:
        size = size * x
    return size * 4

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

image = Image.open('..\\Bear1.jpg')
converter_tensor = transforms.ToTensor()
img_tensor = converter_tensor(image)
img_tensor = img_tensor.unsqueeze(0)
print('\n\n--------Image Input--------')
layer_size = total_size(img_tensor.shape)
print('Data Shape: ',img_tensor.shape, 'Data Size(in bytes):', layer_size)
torch.save(img_tensor, 'TestTensors\img.pt')
out = model(img_tensor)

print('\n--------Begin Split 1--------')

out1 = model.conv1(img_tensor)
layer_size = total_size(out1.shape) #First Split here <--> actually we will just send the image
print('Layer Conv1 -- \t\t Data Shape:', out1.shape, '\t Data Size(in bytes):', layer_size)
torch.save(out1, 'TestTensors\layer1.pt')


out2 = model.bn1(out1)
layer_size = total_size(out2.shape)
print('Layer bn1 -- \t\t Data Shape:', out2.shape, '\t Data Size(in bytes):', layer_size)
torch.save(out2, 'TestTensors\layer2.pt')

out3 = model.relu(out2)
layer_size = total_size(out3.shape)
print('Layer relu -- \t\t Data Shape:', out3.shape, '\t Data Size(in bytes):', layer_size)
torch.save(out3, 'TestTensors\layer3.pt')

out4 = model.maxpool(out3)
layer_size = total_size(out4.shape) #Second split here
print('Layer maxpool -- \t Data Shape:', out4.shape, '\t Data Size(in bytes):', layer_size)
torch.save(out4, 'TestTensors\layer4.pt')
print('\n--------Begin Split 2--------')

out5 = model.layer1(out4)
layer_size = total_size(out5.shape)
print('Layer layer1 -- \t Data Shape:', out5.shape, '\t Data Size(in bytes):', layer_size)
torch.save(out5, 'TestTensors\layer5.pt')

out6 = model.layer2(out5)
layer_size = total_size(out6.shape)
print('Layer layer2 -- \t Data Shape:', out6.shape, '\t Data Size(in bytes):', layer_size)
torch.save(out6, 'TestTensors\layer6.pt')

out7 = model.layer3(out6)
layer_size = total_size(out7.shape) #Third split here
print('Layer layer3 -- \t Data Shape:', out7.shape, '\t Data Size(in bytes):', layer_size)
torch.save(out7, 'TestTensors\layer7.pt')
print('\n--------Begin Split 3--------')

out8 = model.layer4(out7)
layer_size = total_size(out8.shape)
print('Layer layer4 -- \t Data Shape:', out8.shape, '\t Data Size(in bytes):', layer_size)
torch.save(out8, 'TestTensors\layer8.pt')

out9 = model.avgpool(out8)
layer_size = total_size(out9.shape)
print('Layer avgpool -- \t Data Shape:', out9.shape, '\t Data Size(in bytes):', layer_size)
torch.save(out9, 'TestTensors\layer9.pt')

out9 = torch.flatten(out9, 1)
out10 = model.fc(out9)
layer_size = total_size(out10.shape) #Finalized Return Data
print('Layer output(fc) -- \t Data Shape:', out10.shape, '\t\t Data Size(in bytes):', layer_size)
torch.save(out10, 'TestTensors\layer10.pt')
print('\n-----return----')