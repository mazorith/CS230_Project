from socket import *
from struct import pack, unpack
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle

class ClientProtocol:

    def __init__(self):
        self.socket = None
        self.data = None

    def connect(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))

    def close(self):
        self.socket.shutdown(SHUT_WR)
        self.socket.close()
        self.socket = None

    def send_data(self, data):
        data = pickle.dumps(data)
        length = pack('>Q', len(data))

        self.socket.sendall(length)
        self.socket.sendall(data)

        ack = self.socket.recv(1)
        #print(ack)

    def client_handshake(self):
        print('Sending handshake to server')
        self.socket.sendall(b'\00')
        ack = self.socket.recv(1)
        if(ack == b'\00'):
            print('Successfully recived server Handshake')
        else:
            print('Message recived not server ack')

    def handle_data(self):
        collected_message = False
        while not collected_message:
            bs = self.socket.recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.socket.recv(
                    4096 if to_read > 4096 else to_read)

            # send our 0 ack
            assert len(b'\00') == 1
            collected_message = True
            self.socket.sendall(b'\00')
            self.data = pickle.loads(data)

if __name__ == '__main__':

    cp = ClientProtocol()

    cp.connect('127.0.0.1', 55555)
    cp.client_handshake()

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    newModel = torch.nn.Sequential(*list(model.children())[:-5])

    print(newModel)

    image = Image.open('Bear1.jpg')
    img_tensor = transforms.ToTensor(image)
    out = newModel(img_tensor)

    cp.send_data(out)
    cp.handle_data()

    print(cp.data[0])

    cp.close()