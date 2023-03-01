import socket
from struct import pack, unpack
import threading
import sys
import os
import time
import torch
import pickle

slots = 0

#server-side thread function for handling client data
def handle_client(conn, addr, id):
    print(f"[NEW CONNECTION] {addr} connected. -- [THREAD_ID] {id}\n")

    #Large data message handeling loop
    connected = True
    while connected:
        collected_message = False

        while not collected_message:
            print(f"[Thread {id}] Reading data (len)")
            bs = conn.recv(8)
            print(bs)
            (length,) = unpack('>Q', bs)
            data = b''
            print(f"[Thread {id}] Reading data (begin loop)")
            while len(data) < length:
                print(f"[Thread {id}] Reading data (loop)")
                to_read = length - len(data)
                data += conn.recv(
                    500000000 if to_read > 500000000 else to_read)

            collected_message = True

            connected = False
        
        #return data back to client
        print(f"Recived data from thread {id}\n")
        conn.send(bs)
        conn.send(data)
        print(f"Sent Data to client on thread {id}\n")
    conn.close()

#main server thread, will spin up addition message handling thread as they are encountered
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 12345))
    server_socket.listen()
    print("[LISTENING] Server is listening on localhost:12345")

    thread_count = 0
    while thread_count <= 10:
        print('1 Connection passing')
        conn, addr = server_socket.accept()
        thread_count += 1
        thread = threading.Thread(target=handle_client, args=(conn, addr, thread_count))
        thread.start()

    time.sleep(10)

#thread client process of sending and reciving data to server
def start_client(data, id):
    print('starting client')
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))

    print(f"Thread {id} sending: {data.shape}")

    #pack and send data
    data = pickle.dumps(data)
    length = pack('>Q', len(data))
    client_socket.sendall(length)
    client_socket.sendall(data)

    print(f"Thread {id} data sent")

    #Large data message handeling loop
    data, length = None, None
    collected_message = False

    while not collected_message:
        bs = client_socket.recv(8)
        (length,) = unpack('>Q', bs)
        data = b''
        while len(data) < length:
            to_read = length - len(data)
            data += client_socket.recv(
                500000000 if to_read > 500000000 else to_read)

        # send our 0 ack
        collected_message = True
        data = pickle.loads(data)

    print(f"Thread {id} recv:{data.shape}")
    global slots
    slots -= 1

if __name__ == '__main__':
    if sys.argv[1] == 's':
        #all server needs to do is start
        start_server()
    else:
        print('start program')

        #client needs to get data it wants to send over network
        #layer outputs of resnet model in this case I saved before hand
        data_list = []
        data_files = os.listdir('TestTensors\\')
        print(data_files)
        for f in data_files:
            x = torch.load('TestTensors\\' + f)
            data_list.append(x)

        #finally start to send data over network which each one being a thread
        for i, elm in enumerate(data_list):
            print(i, 'thread')
            thread = threading.Thread(target=start_client, args=(elm, data_files[i]))
            started_thread = False
            while not started_thread:
                if slots <= 4:
                    slots += 1
                    thread.start()
                    started_thread = True

        time.sleep(10)
