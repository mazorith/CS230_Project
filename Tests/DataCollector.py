import torch
import torchvision.transforms as transforms
from PIL import Image
import os

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#set layer to extract data
model = torch.nn.Sequential(*list(model.children())[:7]) #up to layer3 layer
print(model)

data_folder_dir = 'J:\\coco2017\\train2017'
output_folder_dir = 'J:\\coco2017\\Res50LayerOuts\\layer3'
lst = os.listdir(data_folder_dir)
lst.sort()

lst2 = os.listdir(output_folder_dir)
lst2.sort()

converter = transforms.ToTensor()
for img_name in lst:
    if (img_name.endswith(".jpg") and not ((img_name[:-4] + '.pt') in lst2)):
        if(len(os.listdir(output_folder_dir)) >= 5000):
            break

        print(img_name)
        img = Image.open(data_folder_dir + '\\' + img_name).convert('RGB')

        img_tensor = converter(img)
        img_tensor = img_tensor.unsqueeze(0)
        #print(img_tensor.shape)
        out = model(img_tensor)

        torch.save(out, output_folder_dir + '\\' + img_name[:-4] + '.pt')