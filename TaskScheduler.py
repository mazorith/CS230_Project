import torch
import torchvision
import thread
import time
import random
from queue import PriorityQueue
from PIL import Image
from CustomRes50 import Resnet50_1, Resnet50_2, Resnet50_3, Resnet50_4
from MasterNode import ClientProtocol

#Currently hardcoding for 3 servers

#task scheduler variables and functions
#these will be multithreaded

def heartbeat_message_thread(client, server_id):
    #loop heartbeat messages
    while heartbeat_termination_flag == 0:
        time.sleep(0.005)
        if server_message_semaphores[server_id] == 0:
            server_message_semaphores[server_id] = 1

            server_heartbeat_message = client.heartbeat(messages_to_heartbeat[server_id], server_id)
            #if server wants to send data it will send a heart message of 2
            if server_heartbeat_message == 2:
                handle_server_message_flags[server_id] = 0

            server_message_semaphores[server_id] = 0

    #once we are finished send heartbeat termination message to kill server process
    #termination message is 255
    if server_message_semaphores[server_id] == 0:
        server_message_semaphores[server_id] = 1

        server_heartbeat_message = client.heartbeat(255, server_id)

        server_message_semaphores[server_id] = 0


def send_message_thread(client, message, server_id):
    message_not_sent = True
    while message_not_sent:
        if server_message_semaphores[server_id] == 0:
            server_message_semaphores[server_id] = 1

            client.send_data(message, server_id)

            server_message_semaphores[server_id] = 0
            message_not_sent = True
        else:
            time.sleep(0.005)

def handle_message_thread(client, server_id):
    message_not_recived = True
    while message_not_recived:
        if server_message_semaphores[server_id] == 0:
            server_message_semaphores[server_id] = 1

            client.handle_data(server_id)
            data = client.sockets_data[server_id]
            
            #TODO: process message here

            server_message_semaphores[server_id] = 0
            message_not_recived = False
        else:
            time.sleep(0.005)

#TODO: change these to priority queues
model2_tasks = []
model3_tasks = []
model4_tasks = []
task_list_semaphores = [0,0,0]

#taskhandler variables and semaphores => 0 are open resources, 1 are held/inactive resources
server_message_semaphores = [0,0,0]
server_model_semaphores = [[(0, 0),(0, 0),(0, 0)],  #server "states"
                           [(0, 0),(0, 0),(0, 0)],  #(is_running, Resnet50_#)
                           [(0, 0),(0, 0),(0, 0)]] 
handle_server_message_flags = [1,1,1]
messages_to_heartbeat = [0,0,0]
heartbeat_termination_flag = 0


#taskhandler client protocall
cp = ClientProtocol()

cp.connect('127.0.0.1', 55555,0)
cp.connect('127.0.0.1', 55556,1)
cp.connect('127.0.0.1', 55557,2)
cp.client_handshake(0)
cp.client_handshake(1)
cp.client_handshake(2)


#client model

client_model = Resnet50_1()

client_model.encoder.load_state_dict(torch.load('AutoEncoders\\encoder_conv1.pt'))
client_model.decoder.load_state_dict(torch.load('AutoEncoders\\decoder_conv1.pt'))

imagenet_data = torchvision.datasets.ImageNet('J:\\ImageNet', split='val')
data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=True)

for data, label in data_loader:
    output = client_model(data)

    #TODO send data to tasks