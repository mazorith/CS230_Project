from multiprocessing import connection
from sqlite3 import connect
import torch
import pickle
import time
import sys
from socket import *
from struct import pack, unpack

class ServerProtocol:

    def __init__(self, is_master):
        #only need one socket for server
        self.socket = None

        #create list of connections, addresses, and storage for message payloads
        self.connections = []
        self.addresses = []
        self.datas = []

        #flags
        self.mutex_flag = 0
        self.is_master = is_master

    #standard listen function
    def listen(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)

    #server handshake function will also serve to accept new connections (will be triggered by task scheduler)
    def server_handshake(self):
        while self.mutex_flag != 0:
            time.sleep(0.001)
        
        self.mutex_flag = 1

        conns = 3
        if self.is_master:
            conns =4 

        for i in range(conns):
            (connection, addr) = self.socket.accept()
            self.connections.append(connection)
            self.addresses.append(addr)
            self.datas.append('')

            handshake = connection.recv(1)
            if(handshake == b'\00'):
                print('Successfully recived client Handshake')
                print('Sending handshake ack')
                connection.sendall(b'\00')
            else:
                print('Message recived not client handshake')

            if self.is_master:
                break

        self.mutex_flag = 0

        #for task handling purposes return the new addresses together 
        return self.addresses[-4:] if is_master else self.addresses[-3:]

    #only need to handle data with respect to specific addresses 
    def handle_data(self, addr):
        idx = self.addresses.index(addr)

        collected_message = False
        while not collected_message:
            bs = self.connections[idx].recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.connections[idx].recv(
                    50000 if to_read > 50000 else to_read)

            # send our 0 ack
            assert len(b'\00') == 1
            collected_message = True
            self.connections[idx].sendall(b'\00')
            self.datas[idx] = pickle.loads(data)
        
    #Same as above we only need to send with respect to specific addresses as we will be multithreading
    def send_data(self, data, addr):
        data = pickle.dumps(data)
        length = pack('>Q', len(data))
        idx = self.addresses.index(addr)

        self.connections[idx].sendall(length)
        self.connections[idx].sendall(data)

        ack = self.connections[idx].recv(1)
        #print(ack)

    #basic heartbeat function
    def heartbeat(self, return_message, addr):
        idx = self.addresses.index(addr)

        recv_message = self.connections[idx].recv(1)
        recv_message = int.from_bytes(recv_message, 'big')

        message = return_message.to_bytes(1, 'big')
        self.connections[idx].sendall(message)

        return recv_message

    def close(self):
        for i in range(len(self.connections)):
            #self.connections[i].shutdown(SHUT_WR)
            self.connections[i].close()

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