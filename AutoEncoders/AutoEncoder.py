import torch
import torch.nn as nn
import torchvision
import os
import random

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer1 = nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["layer1_shape"])
        self.encoder_hidden_layer2 = nn.Linear(in_features=kwargs["layer1_shape"], out_features=kwargs["layer2_shape"])
        self.encoder_output_layer = nn.Linear(in_features=kwargs["layer2_shape"], out_features=kwargs["output_shape"])

    def forward(self, features):
        activation = self.encoder_hidden_layer1(features)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decoder_hidden_layer1 = nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["layer1_shape"])
        self.decoder_hidden_layer2 = nn.Linear(in_features=kwargs["layer1_shape"], out_features=kwargs["layer2_shape"])
        self.decoder_output_layer = nn.Linear(in_features=kwargs["layer2_shape"], out_features=kwargs["output_shape"])

    def forward(self, features):
        activation = self.decoder_hidden_layer1(features)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


def total_size(shape):
    size = 1
    for x in shape:
        size = size * x
    return size



if __name__ == '__main__':
    model1 = Encoder(input_shape=1024, layer1_shape=512, layer2_shape=128, output_shape=64)
    model2 = Decoder(input_shape=64, layer1_shape=128, layer2_shape=512, output_shape=1024)
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    layer = 'layer3'

    output_folder_dir = 'J:\\coco2017\\Res50LayerOuts\\' + layer
    lst = os.listdir(output_folder_dir)
    lst.sort()
    lst = lst[:5000]

    loss_file = open(layer + '_loss.txt', 'w')

    try:
        epochs = 4
        for epoch in range(epochs):
            loss = 0
            i=0
            random.shuffle(lst)
            for data_name in lst:
                i += 1

                data = torch.load(output_folder_dir + '\\' + data_name)
                og_shape = data.shape
                data = data.view(-1, 1024)
                
            
                optimizer.zero_grad()
                output = model1(data)

                new_shape = output.shape

                output = model2(output)

                train_loss = criterion(output, data)
                train_loss.backward()
                optimizer.step()

                loss += train_loss.item()

                loss_file.write(str(loss/i)+'\n')

                if i%25 == 0:
                    print("epoch : {}, data : {}, loss = {:.6f}".format(epoch + 1, i, (loss/i)))
                    print("Orginal Shape & size : {}->{}, New Shape & size : {}->{}".format(og_shape, total_size(og_shape), new_shape, total_size(new_shape)))

            loss = loss/i
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

        torch.save(model1.state_dict(), os.getcwd() + '\\encoder_' + layer + '.pt')
        torch.save(model2.state_dict(), os.getcwd() + '\\decoder_' + layer + '.pt')
        loss_file.close()
    except:
        torch.save(model1.state_dict(), os.getcwd() + '\\encoder_' + layer + '.pt')
        torch.save(model2.state_dict(), os.getcwd() + '\\decoder_' + layer + '.pt')
        loss_file.close()
    