from multiprocessing import connection
from sqlite3 import connect
import torch
import pickle
import time
import sys
from socket import *
from struct import pack, unpack

class ServerProtocol:

    def __init__(self):
        self.socket = None
        self.connection = None
        self.data = None

    def listen(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)

    def handle_data(self):
        collected_message = False
        while not collected_message:
            bs = self.connection.recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.connection.recv(
                    4096 if to_read > 4096 else to_read)

            # send our 0 ack
            assert len(b'\00') == 1
            collected_message = True
            self.connection.sendall(b'\00')
            self.data = pickle.loads(data)
        
    def server_handshake(self):
        (connection, addr) = self.socket.accept()
        self.connection = connection
        handshake = self.connection.recv(1)
        if(handshake == b'\00'):
            print('Successfully recived client Handshake')
            print('Sending handshake ack')
            connection.sendall(b'\00')
        else:
            print('Message recived not client handshake')

    def send_data(self, data):
        data = pickle.dumps(data)
        length = pack('>Q', len(data))

        self.connection.sendall(length)
        self.connection.sendall(data)

        ack = self.connection.recv(1)
        #print(ack)

    def close(self):
        self.connection.shutdown(SHUT_WR)
        self.connection.close()
        self.socket.close()
        self.socket = None

#main functionality for testing/debugging
if __name__ == '__main__':
    sp = ServerProtocol()
    sp.listen('127.0.0.1', 55555)
    sp.server_handshake()

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    newModel = torch.nn.Sequential(*list(model.children())[5:])
    newModel = torch.nn.Sequential(*list(newModel.children())[:4])
    last_layer = torch.nn.Sequential(*list(model.children())[9:])

    sp.handle_data()

    print('Handled Data')
    print(sp.data.shape)
    out = newModel(sp.data)
    out = torch.flatten(out, 1)
    out = last_layer(out)
    print('success, sending back output')

    sp.send_data(out)
    sp.close()