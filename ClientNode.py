from socket import *
from struct import pack, unpack
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle
import time

class ClientProtocol:

    def __init__(self):
        #for each server connection we will define a client socket
        self.sockets = []
        self.sockets_data = []

        #flags
        self.mutex_flag = 0

    #connection function will need to be called before client handshake
    def connect(self, server_ip, server_port=5555):
        while self.mutex_flag != 0:
            time.sleep(0.001)

        self.mutex_flag = 1
        self.sockets.append(socket(AF_INET, SOCK_STREAM))
        self.sockets[len(self.sockets)-1].connect((server_ip, server_port))
        self.sockets_data.append('')
        self.mutex_flag = 0

        return len(self.sockets) - 1

    #client handshake needs to be called immediately after connect function 
    def client_handshake(self, socket_number):
        print('Sending handshake to server')
        self.sockets[socket_number].sendall(b'\00')
        ack = self.sockets[socket_number].recv(1)
        if(ack == b'\00'):
            print('Successfully recived server Handshake')
        else:
            print('Message recived not server ack')

    #as we are multithreading we only need to work about sending data through one connection
    def send_data(self, data, socket_number):
        data = pickle.dumps(data)
        length = pack('>Q', len(data))

        self.sockets[socket_number].sendall(length)
        self.sockets[socket_number].sendall(data)

        ack = self.sockets[socket_number].recv(1)
        #print(ack)

    #same logic as above function, we only need to recive data from on connection
    def handle_data(self, socket_number):
        collected_message = False
        while not collected_message:
            bs = self.sockets[socket_number].recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.sockets[socket_number].recv(
                    500000000 if to_read > 500000000 else to_read)

            # send our 0 ack
            assert len(b'\00') == 1
            collected_message = True
            self.sockets[socket_number].sendall(b'\00')
            self.sockets_data[socket_number] = pickle.loads(data)

    #heartbeat will only send/recv a single byte 
    def heartbeat(self, message_number, socket_number):
        message = message_number.to_bytes(1,'big')
        
        self.sockets[socket_number].sendall(message)
        ack_message = self.sockets[socket_number].recv(1)

        return int.from_bytes(ack_message, 'big')

    def close(self, socket_number = -1):
        if socket_number == -1:
            for socket in self.sockets:
                #socket.shutdown(SHUT_WR)
                socket.close()
                socket = None
        else:
            #self.sockets[socket_number].shutdown(SHUT_WR)
            self.sockets[socket_number].close()
            self.sockets[socket_number] = None

#main functionality for testing/debugging
if __name__ == '__main__':

    cp = ClientProtocol()

    cp.connect('127.0.0.1', 55555,0)
    cp.connect('127.0.0.1', 55556,1)
    cp.connect('127.0.0.1', 55557,2)
    
    cp.client_handshake(0)
    cp.client_handshake(1)
    cp.client_handshake(2)

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # newModel = torch.nn.Sequential(*list(model.children())[:-5])

    # image = Image.open('Bear1.jpg')
    # converter_tensor = transforms.ToTensor()
    # img_tensor = converter_tensor(image)
    # img_tensor = img_tensor.unsqueeze(0)
    # print(img_tensor.shape)
    # out = newModel(img_tensor)

    # print(out.shape)
    # cp.send_data(out)
    # cp.handle_data()

    # print(cp.data[0])

    # cp.close()