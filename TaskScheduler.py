import torch
import torchvision
import threading
import sys
import time
import random
from queue import PriorityQueue
from PIL import Image
from ClientNode import ClientProtocol
from ServerNode import ServerProtocol

'''
=====================================================================================================
For simplicity this implementaion is assuming only one node has tasks to preform (the master node).

Also for further simplicity, this will be hardcode for a 4 node setup, including the master node, 
since we are limiting the resource slot capacity per worker node to 3. 

We will also be using threaded sockets with global variables to share data between threads
=====================================================================================================
'''

is_master = False #using as switch variable to control functionality
all_task_completed = False #global completion variable

#connection lists
cp_list = []
#this will be a list of lists representing the server connections per device
sp_addr_list = []

#these will be lists of 3 one for connection on each cp and sp
cp_connection_semaphores = []
sp_connection_semaphores = []

heartbeat_messages = [0] #will be a list of 1 for nodes, will expand for connections on master node

#only for master node will be a list of lists of tuples #(is_running, model_split) for each node in network
node_resource_semaphores = [] 

#lists containing the data inputs for the model
#TODO: change lists to priority queues for time squental processing 

tasks_to_offload = []
raw_img_tasks = []

#these tasks will be lists of tuples (node, #_of_tasks) for master node
#else they will be lists of actual intermidate values for resource nodes
split2_tasks = []
split3_tasks = []
return_tasks = [] 

#TODO: add complexity to completed_tasks
completed_tasks = (0,0) #currently (# of tasks complete, # of tasks correct [top 1%])

#list of models splits for the node, will contain 1 for master and 3 for worker nodes
resources = []
resource_semaphores = []
#min_model_exc_time = 0.82
avg_model_exc_time = 1.08
avg_batch_model_offload_time = [2.16, 2.30, 1.87] #from calculations using recorded data - batch of [3,6,6]
avg_offload_factors = [0.76,0.73,0.36] #customizable modification factors for avg_batch_model_offload_time

def server_connection_thread(sp):
    global is_master
    global sp_addr_list
    global node_resource_semaphores

    #TODO: add functionality to start heartbeat messages??


    #TODO: add flag so we do not append more node resouces while scheduler is searching for nodes
    if is_master:
        while is_master:
            ret_list = sp.server_handshake()
            sp_addr_list.append(ret_list)

            #optimal split configurations without dynamic contex switching
            #even though we are hardcoding 3 worker nodes the same algorithm can be used here for more just using (%1 & %3) else (%2)
            num_of_nodes = len(sp_addr_list)
            if num_of_nodes == 1 or num_of_nodes == 3:
                #default split (1,2,3) || tri-split (1,2,3) - (1,2,2) - (2,2,3)
                node_resource_semaphores.append([(1,0),(2,0),(3,0)])
                pass
            elif num_of_nodes == 2:
                #2 node split (1,2,2) - (2,2,3)
                #TODO: call a context switch of node 1
                node_resource_semaphores.append([(2,0),(2,0),(3,0)])
                pass 

    #currently only master node cares about external node resource states
    else:
        ret_list = sp.server_handshake()
        sp_addr_list.append(ret_list)

#establish connection to server protocol (will not be used by master node)
def client_connection_thread(cp, addr, port, is_first_connection):
    global cp_list
    global cp_connection_semaphores

    iter_var = 4 if is_first_connection else 3
    cp_append_list = []
    semaphore_list = []
    
    for i in range(iter_var):
        slot = cp.connect(addr, port)
        cp_append_list.append(slot)
        semaphore_list.append(0)
        cp.client_handshake(slot)
        
    cp_list.append(cp_append_list)
    cp_connection_semaphores.append(semaphore_list)

#personal nodes resource will be element 0 in resources
def personal_execution(features, labels):
    global resources
    global resources_semaphores
    global completed_tasks

    resources[0](features) 

    #TODO:create code for checking accuracy

    completed_tasks[0] += 1
    resources_semaphores[0] = 0

#thread function for master node only
def scheduler():
    global all_task_completed
    global node_resource_semaphores
    global split2_tasks
    global split3_tasks 
    global return_tasks 

    avg_message_delays = [0.38, 0.34, 0.42, 0.02] #raw_image, split-1 out, split-2 out, final output

    #main scheduler loop
    while not all_task_completed:
        
        #if  we have tasks waiting to be offloaded check through node resources
        for index, node in enumerate(node_resource_semaphores):
            for slot in node:
                #check if avaible resource for raw image
                if len(tasks_to_offload) > 0:
                    if slot[1] == 0:
                        tasks_to_offload.pop(0)
                        #TODO:send message
                        break
                
                #check if avaible resouce for second split tasks
                if slot[0] == 2 and slot[1] == 0:
                    for elm in split2_tasks:
                        if elm[1] >= 0:
                            elm[1] -= 1
                            #if elm[0] == index:
                                #TODO: send message to node to route data to its own slot
                           # else:
                                #TODO:send message to node to send data to elm[0] node
                                #TODO:Tell elm[0] node to listen and recive data coming from node for slot

                #check if avaible resource for third split tasks
                if slot[0] == 3 and slot[1] == 0:
                    for elm in split3_tasks:
                        if elm[1] >= 0:
                            elm[1] -= 1
                            #if elm[0] == index:
                                #TODO: send message to node to route data to its own slot
                           # else:
                                #TODO:send message to node to send data to elm[0] node
                                #TODO:Tell elm[0] node to listen and recive data coming from node for slot

        #check for outputs waiting on return
        for elm in return_tasks:
            if elm[1] >= 0:
                elm[1] -= 1
                #TODO: request message from node to route result data back to master

        time.sleep(0.01)

#this will be use on every the first of every 4 connections of master or the first client protocol socket of every node
def heartbeat(protocol, protocol_index):
    global cp_list
    global cp_connection_semaphores
    global sp_addr_list
    global sp_connection_semaphores
    global is_master
    global all_task_completed
    global heartbeat_messages
    
    if isinstance(protocol, ClientProtocol):
        while not all_task_completed:
            protocol.heartbeat(heartbeat_messages[0], cp_list[protocol_index][0])
            time.sleep(0.01)
    else:
        while not all_task_completed:
            protocol.heartbeat(heartbeat_messages[protocol_index], sp_addr_list[protocol_index][0])

#this is only for sending data over channel                
def send_message(protocol, protocol_index, data, addr=None):
    global cp_list
    global cp_connection_semaphores
    global sp_addr_list
    global sp_connection_semaphores
    global is_master

    #this is basic logic, TODO: create a secondary rule to prevent collision
    if isinstance(protocol, ClientProtocol):
        #conn_offset = cp_list[protocol_index]
        offset = 4 if is_master else 3 

        #while
    else:
        conn_list = sp_addr_list[protocol_index]
        channel_selection = 0
        iter_offset = 4 if is_master else 3

        avalible_channel = False
        while not avalible_channel:
            for i in range(iter_offset):
                time.sleep(0.01)

                if iter_offset != 4 and i != 0 and sp_connection_semaphores[protocol_index][i] == 0:
                    sp_connection_semaphores[protocol_index][i] = 1
                    if not is_master:
                        continue
                        #TODO:send message to master that server port is ready
                        #Set some variable to true while we wait for response from server to ensure that connection isn't processing
                        #have another variable which will 
                    
                    #need to encapulate below this in case we there is a resource conflict
                    channel_selection = i
                    avalible_channel = True
                    break
                    

        protocol.send_data(data, conn_list[protocol_index][channel_selection])

        sp_connection_semaphores[protocol_index][channel_selection] = 0


def receive_message(protocol):

if __name__ == '__main__':
    #TODO: refactor to have master ip as input of sys.argv[2], and port as sys.argv[3]
    if sys.argv[1] == '--master': 
        print('starts as master')

        #set inital variables
        is_master = True
        thread_list = []

        #spin up server protocol
        sp = ServerProtocol()
        sp.listen('127.0.0.1', 5555)
        thread_list.append(threading.Thread(target=server_connection_thread, args=(sp)))

        #set resources/load model and dataset

        resources.append(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True))
        resources_semaphores = 0

        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((500,500)),
                                                torchvision.transforms.ToTensor()])
        imagenet_data = torchvision.datasets.ImageNet('J:\\ImageNet', transform = transform, split='val')
        data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=False) 

        #task generation loop
        frame_rate = 0.0334 #simulated frame rate timing (0.0334 ~ 29.97 fps)

        personal_task_start_time = 0
        task_num = 0 #also iteration count
        personal_task_processing_number = 0
        last_personal_task_process = 1 #begining of last batch of personal tasks

        current_num_tasks_offloaded = 0

        #Spin up schdeuler thread 
        thread_list.append(threading.Thread(target=scheduler, args=()))

        with torch.no_grad():
            for features, labels in data_loader:
                frame_capture_start = time.time()

                task_num += 1
                raw_img_tasks.append((task_num, features, labels))

                #if personal resource is avaible send oldest frame saved to personal resource 
                if resource_semaphores[0] == 0:  
                    task = raw_img_tasks.pop(0)
                    personal_task_processing_number = task[0]

                    personal_task_start_time = time.time()
                    threading.Thread(target=personal_execution, args=(task[1], task[2]))

                    resource_semaphores[0] = 1

                #if there are other nodes in the network and queued tasks decide if newest task should be offloaded
                if len(sp_addr_list) > 0 and len(raw_img_tasks) > 0:
                    #estimate time need to run the personal resource including current execution
                    personal_time_est = (task_num - last_personal_task_process - 1)*avg_model_exc_time
                    personal_time_est += (avg_model_exc_time - (time.time() - personal_task_start_time)) 
                    
                    #get calulated single task estimates
                    network_config = len(sp_addr_list)-1
                    offload_time_est = avg_batch_model_offload_time[network_config]

                    #depending on load of the system add additional delay(may need to remove /2 for avg_offload_factors)
                    if current_num_tasks_offloaded >= 3 and network_config == 0:
                        offload_time_est += (current_num_tasks_offloaded - 2) * (avg_offload_factors[network_config]/2)
                    elif current_num_tasks_offloaded >= 6 and network_config > 0:
                        offload_time_est += (current_num_tasks_offloaded - 5) * (avg_offload_factors[network_config]/2)
                    
                    #if offload estimate is less that personal execution estimate offload task
                    if offload_time_est <= personal_time_est:
                        #TODO: edit this offloaded var for global
                        current_num_tasks_offloaded += 1

                        #TODO: call thread to send task to first offload machine
                        tasks_to_offload.append(raw_img_tasks.pop())

                #simulate a framerate 
                if frame_rate > time.time() - frame_capture_start:
                    time.sleep(frame_rate - (time.time() - frame_capture_start))

            #TODO:Join task Threads
            #all_task_completed = True

    else:
        print('starts as worker')

        #set inital variables
        #TODO: cp_list will store connections in for of (ip, port) -- refactor to commandline args for master node
        #cp_list.append((sys.argv[2]), int(sys.argv[3]))
        cp_list.append(('127.0.0.1', 5555))
        thread_list = []

        cp = ClientProtocol()

        for i in range(4):
            cp.connect('127.0.0.1', 5555)
            cp.client_handshake(i)

        #get initial message from master node
        #if first node server protocol not needed else create server protocol
        
        #once connections are established wait for master to direct resource slots
        #simply loop heartbeat and functionality threads until termination heartbeat is sent by master 
        