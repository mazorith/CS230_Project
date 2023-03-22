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

#these will be a list of 1 for nodes, will expand for connections on master node
heartbeat_messages = [[0,0,0,0,0]] 
received_heartbeat_messages = [0]
heartbeat_semaphores = [0]

#TODO:will explain this later
ready_node_connection_list = []

#only for master node will be a list of lists of #[is_running, model_split] for each node in network
node_resource_semaphores = [] 
#as above only for master node will be a list of lists of #[connecting_node, protocol_type, is_open], ...]
node_channel_semaphores = []

#lists containing the data inputs for the model
#TODO: change lists to priority queues for time squental processing 

tasks_to_offload = []
raw_img_tasks = []

#these tasks will be lists of lists (node, #_of_tasks) for master node
#else they will be lists of actual intermidate values for resource nodes
split2_tasks = []
split3_tasks = []
return_tasks = [] 

completed_tasks = [0,0] #currently (# of tasks complete, # of tasks correct [top 1%])

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
    #TODO: append a (0,0) to ready_node_connection_list
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

#takes a list of splits as [1,2,3] or [2,2,3]
def set_model_splits(splits):
    global resources
    global resource_semaphores

    for split in splits:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        if split = 1:
            sub_model = torch.nn.Sequential(*list(model.children())[:4])
            resources.append(sub_model)

        elif split ==2:
            sub_model = torch.nn.Sequential(*list(model.children())[4:7])
            resources.append(sub_model)

        elif split == 3:
            sub_model = torch.nn.Sequential(*list(model.children())[7:9])
            last_layer = torch.nn.Sequential(*list(model.children())[9:])
            resources.append([sub_model, last_layer])

        resource_semaphores.append([split, 0])


#personal nodes resource will be element 0 in resources
#model split = 0 if full model, = 1 for split 1, = 2 for split 2, and so on
def personal_execution_thread(features, labels, resources_index, model_split=0):
    global resources
    global resources_semaphores
    global completed_tasks
    global is_master

    global split2_tasks
    global split3_tasks
    global return_tasks

    if model_split == 3:
        output = resources[resources_index](features) 
    else:
        output = resources[resources_index][0](features)
        output = torch.flatten(out, 1)
        output = resources[resources_index][1](features)

    if is_master:
        if (torch.argmax(output, axis=1)==labels):
            completed_tasks[1] += 1
        completed_tasks[0] += 1
    else:
        if model_split == 1:
            split2_tasks.append(output)
        elif model_split == 2:
            split3_tasks.append(output)
        elif model_split == 3:
            output = torch.argmax(output, axis=1)
            return_tasks.append(output)
    
    resources_semaphores[resources_index] = 0

#thread function for master node only
def scheduler_thread():
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
#secondary protocol is needed for nodes sp node
def heartbeat_thread(protocol, protocol_index, secondary_protocol=None): 
    global cp_list
    global cp_connection_semaphores
    global sp_addr_list
    global sp_connection_semaphores
    global is_master
    global all_task_completed
    global heartbeat_messages
    global heartbeat_semaphores
    global received_heartbeat_messages

    global resources_semaphores
    global node_channel_semaphores

    global ready_node_connection_list

    global split2_tasks
    global split3_tasks
    global return_tasks 
    
    #master to node
    # 0 basic connection do nothing
    # 1 master will be sending data of other nodes for connection to new node, ready any master port
    # 2 master telling node to ready itself for new node connection
    # 3 master is directing traffic, ready first master port for infromation (data for split 2 and 3)
    # 4 master will be sending raw image data to node, ready any master port (data for split 1)
    #   1, 3, and 4 will be followed by a connection port(s) in which to send data
    # 255 master is terminating connection (have node kill itself)

    #node to master
    # 0 basic connection do nothing
    # 5 node wants to send completion data to master
    # 6 node finished a task and wants to send state to master, request for port (output for split 3)
    # 255 confirmation on termination (will kill itself)

    if isinstance(protocol, ClientProtocol):
        while not all_task_completed:
            recived = protocol.heartbeat(heartbeat_messages[0][0], cp_list[protocol_index][0])
            #for 1-4 we will need to send another heartbeat back as confirmation of request

            #network updates
            if recived == 1:
                heart_check = protocol.heartbeat(1, cp_list[protocol_index][0])
                if heart_check == 1:
                    protocol.handle_data(cp_list[protocol_index][0])

                    #TODO:recive message to set split contexts (to bypass need to create a function for it)

                    #recive list of addrs to connect too
                    protocol.handle_data(cp_list[protocol_index][0])
                    addrs = protocol.sockets_data[protocol_index]

                    #connect to them
                    for addr in addrs:
                        threading.Thread(target=client_connection_thread, args=(protocol, addr[0], 5555, False))

            if recived == 2:
                heart_check = protocol.heartbeat(2, cp_list[protocol_index][0])
                if heart_check == 2:
                    #listen for a new connection to be made
                    threading.Thread(target=server_connection_thread, args=(secondary_protocol))
                

            #intermediate data transfer
            if recived == 3:
                heart_check = protocol.heartbeat(3, cp_list[protocol_index][0])
                if heart_check == 3:
                    #revice state info
                    protocol.handle_data(cp_list[protocol_index][0])
                    data=protocol.sockets_data[protocol_index]

                    #sending
                    if data[0] == 0: 
                        data_to_send = split1_tasks.pop(0) if data[1] == 0 else split2_tasks.pop(0)
                        threading.Thread(target=send_message_thread, args=(protocol if data[2] == 0 else secondary_protocol, 
                                                                                data_to_send, data[3][0], data[3][1]))
                    #reciving
                    elif data[0] == 1: 
                        threading.Thread(target=receive_message_thread, args=(protocol if data[2] == 0 else secondary_protocol, 
                                                                                data[3][0], data[3][1], data[1]+1))

            if recived == 4:
                heart_check = protocol.heartbeat(4, cp_list[protocol_index][0])
                if heart_check == 4:
                    #recive state info
                    protocol.handel_data(cp_list[protocol_index][0])
                    addr = protocol.sockets_data[protocol_index]

                    #recv raw_image
                    cp_connection_semaphores[addr[0]][addr[1]] = 1
                    threading.Thread(target=receive_message_thread, args=(protocol, addr[0], addr[1]))

            #output transfer and state updates
            if recived == 5:
                #since we will return the argmaxed result message time is negligible
                return_data = return_tasks.pop(0)
                protocol.send_data(return_data, cp_list[protocol_index][0])
                heartbeat_messages[0][0] = 0

            if recived == 6:
                #return a state list of resource flags, cp flags (without master), and sp flags
                protocol.send_data([resources_semaphores, cp_connection_semaphores[1:], sp_connection_semaphores], cp_list[protocol_index][0])
                heartbeat_messages[0][0] = 0
                
            if recived == 255:
                #TODO:die
                heartbeat_messages[0][0] = 0

            if recived != 0:
                received_heartbeat_messages[0][0] = recived 

            time.sleep(0.005)

    elif isinstance(protocol, ServerProtocol):
        while not all_task_completed:
            recived = protocol.heartbeat(heartbeat_messages[protocol_index][0], sp_addr_list[protocol_index][0])
            #we will only care about node requests if master node is at idle with the node
            #we will need to send a heartbeat back as confirmation of request
            if heartbeat_messages[protocol_index][0] == 0:
                if recived == 5:
                    heart_check = protocol.heartbeat(5, sp_addr_list[protocol_index][0])
                    if heart_check == 5:
                        #recive task completion data
                        protocol.handle_data(sp_addr_list[protocol_index][0])
                        idx = protocol.addresses.index(sp_addr_list[protocol_index][0])
                        state_data = protocol.datas[idx]

                        #add to stats
                        completed_tasks[0] += 1
                        if data == labels:
                            completed_tasks[1] += 1

                if recived == 6:
                    heart_check = protocol.heartbeat(6, sp_addr_list[protocol_index][0])
                    if heart_check == 6:
                        #recive state data
                        protocol.handle_data(sp_addr_list[protocol_index][0])
                        idx = protocol.addresses.index(sp_addr_list[protocol_index][0])
                        state_data = protocol.datas[idx]

                        #update resource states
                        for i, data in enumerate(state_data[0]):
                            node_resource_semaphores[protocol_index][i] = data

                        #update channel states
                        node_channel_semaphores[protocol_index] = state_data[1:]

            #network updates
            if recived == 1:
                connection_list = []
                for i, l in enumerate(sp_addr_list):
                    if i != len(sp_addr_list) - 1:
                        connection_list.append(l[0][0])
                        ready_node_connection_list[i] = 0
                    

                protocol.send_data(connection_list, sp_addr_list[protocol_index][0])

                #TODO:send message to set split contexts of node (to bypass need to create a function for it)
                
                heartbeat_messages[protocol_index][0] = 0

            if recived == 2:
                ready_node_connection_list[protocol_index] = 1
                heartbeat_messages[protocol_index][0] = 0

            #intermediate data transfer
            if recived == 3:
                data = heartbeat_messages[protocol_index][1:]
                protocol.send_data(data, sp_addr_list[protocol_index][0])
                heartbeat_messages[protocol_index] = [0,0,0,0,0]

            if recived == 4:
                data = heartbeat_messages[protocol_index][1]
                protocol.send_data(data, sp_addr_list[protocol_index][0])
                
                threading.Thread(target=send_message_thread, args=(protocol, heartbeat_messages[protocol_index][3], 
                                                                             heartbeat_messages[protocol_index][2][0], 
                                                                             heartbeat_messages[protocol_index][2][1]))

                heartbeat_messages[protocol_index] = [0,0,0,0,0]

            if recived != 0:
                received_heartbeat_messages[protocol_index][0] = recived  
            
            time.sleep(0.005)

#this is only for sending data over channel                
def send_message_thread(protocol, data, protocol_index, channel_selection=0):
    global cp_list
    global cp_connection_semaphores
    global sp_addr_list
    global sp_connection_semaphores

    if isinstance(protocol, ClientProtocol):
        protocol.send_data(data, cp_list[protocol_index][channel_selection])

        #this should be set to 1 before calling this thread function
        cp_connection_semaphores[protocol_index][channel_selection] = 0

    elif isinstance(protocol, ServerProtocol):
        protocol.send_data(data, sp_addr_list[protocol_index][channel_selection])

        #this should be set to 1 before calling this thread function
        sp_connection_semaphores[protocol_index][channel_selection] = 0

    #TODO:if not is master update heartbeat message to update state

#this is only for receiving data over channel  
def receive_message_thread(protocol, protocol_index, channel_selection=0, data_type=0):
    global cp_list
    global cp_connection_semaphores
    global sp_addr_list
    global sp_connection_semaphores
    
    global raw_img_tasks 
    global split2_tasks 
    global split3_tasks 

    if isinstance(protocol, ClientProtocol):
        protocol.handle_data(cp_list[protocol_index][channel_selection])
        data = protocol.sockets_data[cp_list[protocol_index][channel_selection]]

        if data_type == 0:
            raw_image_tasks.append(data)
        elif data_type == 1:
            split2_tasks.append(data)
        elif data_type == 2:
            split3_tasks.append(data)

        #this should be set to 1 before calling this thread function
        cp_connection_semaphores[protocol_index][channel_selection] = 0

    elif isinstance(protocol, ServerProtocol):
        protocol.handle_data(sp_addr_list[protocol_index][channel_selection])

        indx = protocol.addresses.index(sp_addr_list[protocol_index][channel_selection])
        data = protocol.datas[indx]

        elif data_type == 1:
            split2_tasks.append(data)
        elif data_type == 2:
            split3_tasks.append(data)
        elif data_type == 3:
            completed_tasks[0] += 1
            if data == labels:
                completed_tasks[1] += 1

        #this should be set to 1 before calling this thread function
        sp_connection_semaphores[protocol_index][channel_selection] = 0

    #TODO:if not is master update heartbeat message to update state

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
        thread_list.append(threading.Thread(target=scheduler_thread, args=()))

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
                    #TODO:Edit this for new personal exc funtionality
                    threading.Thread(target=personal_execution_thread, args=(task[1], task[2]))

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
        