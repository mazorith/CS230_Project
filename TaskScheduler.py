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


P.S: Yes the logic in the code is messy, read at your own risk. >.>
=====================================================================================================
'''

'''
                                        ---------------------------
----------------------------------------| Global shared variables |----------------------------------------
                                        ---------------------------
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
heartbeat_semaphores = [0,0] #these are for worker nodes, if set they want to hearbeat either 6 or 7 respectively 

#only for master node will be a list of lists of #[is_running, model_split] for each node in network
node_resource_semaphores = [] 
#as above only for master node will be a list of lists of #[connecting_node, protocol_type, is_open], ...]
node_channel_semaphores = []

#lists containing the data inputs for the model
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

#this is for nodes context switching
splits = [1,2,3]
context_switch_semaphore = 0

#these are recorded average times for decision making and time "cushion" factors
#min_model_exc_time = 0.82
avg_model_exc_time = 1.08
avg_batch_model_offload_time = [2.16, 2.30, 1.87] #from calculations using recorded data - batch of [3,6,6]
avg_offload_factors = [0.76,0.73,0.36] #customizable modification factors for avg_batch_model_offload_time

'''
                                        -------------------------------
----------------------------------------| Connection thread functions |----------------------------------------
                                        -------------------------------
'''

def server_connection_thread(sp):
    global is_master
    global sp_addr_list
    global sp_connection_semaphores
    global node_resource_semaphores
    global heartbeat_messages

    global split2_tasks
    global split3_tasks

    heartbeat_threads = []

    #TODO: add flag so we do not append more node resouces while scheduler is searching for nodes
    if is_master:
        while len(sp_addr_list) < 3: #hardcoding for 3 nodes
            ret_list = sp.server_handshake()

            #append basic lists
            sp_addr_list.append(ret_list)
            sp_connection_semaphores.append([0,0,0,0])
            split2_tasks.append(0)
            split3_tasks.append(0)

            #ready old nodes for new node connection
            if len(heartbeat_messages) > 0:
                for i in range(len(heartbeat_messages)):
                    while heartbeat_messages[i][0] != 0:
                        time.sleep(0.001)
                    heartbeat_messages[i][0] = 2

            #initialize heartbeat message for receiving existing nodes addresses to connect too
            heartbeat_messages.append([1,0,0,0,0])

            #optimal split configurations without dynamic contex switching
            #even though we are hardcoding 3 worker nodes the same algorithm can be used here for more nodes just using (%1, %2, & %3)
            num_of_nodes = len(sp_addr_list)
            if num_of_nodes == 1:
                #default split (1,2,3) 
                node_resource_semaphores.append([[1,0],[2,0],[3,0]])
            elif num_of_nodes == 2:
                #2 node split (1,2,2) - (2,2,3)
                node_resource_semaphores.append([[2,0],[2,0],[3,0]])
            elif num_of_nodes == 3:
                #tri-split (1,2,3) - (1,2,2) - (2,2,3)

                #do a context switch on node 2 before appending node 3
                node_resource_semaphores[1] = [[1,0],[2,0],[2,0]]
                while heartbeat_messages[1][0] != 0:
                    time.sleep(0.001)
                heartbeat_messages[1][0] = 3

                node_resource_semaphores.append([[2,0],[2,0],[3,0]])


            new_thread = threading.Thread(target=heartbeat_thread, args=(sp, len(sp_addr_list)-1)).start()
            heartbeat_threads.append(new_thread)

        for thread in heartbeat_threads:
            thread.join()

    #currently only master node cares about external node resource states
    else:
        ret_list = sp.server_handshake()
        sp_addr_list.append(ret_list)
        sp_connection_semaphores.append([0,0,0,0])
    

#establish connection to server protocol (will not be used by master node)
#we need to pass in cp so we can spin up the heartbeat thread for client
def client_connection_thread(cp, addr, port, is_first_connection, sp=None):
    global cp_list
    global cp_connection_semaphores
    global heartbeat_semaphores

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

    if is_first_connection:
        new_thread = threading.Thread(target=heartbeat_thread, args=(cp, len(cp_list)-1, sp)).start()
        if new_thread != None:
            new_thread.join()

    #try to send updated state to master
    heartbeat_semaphores[1] = 1


'''
                               ------------------------------------------------
-------------------------------| model creation and execution thread function |--------------------------------
                               ------------------------------------------------
'''

#takes a list of splits as [1,2,3] or [2,2,3]
def set_model_splits(splits):
    global resources
    global resource_semaphores
    global heartbeat_semaphores

    processes_empty = False

    #ensure that all tasks have been processed before context switching
    while not processes_empty:
        processes_empty = True
        for resource in resources:
            if resources[1] != 0:
                processes_empty = False
        if not processes_empty:
            time.sleep(0.01)

    #context switch/set splits
    for split in splits:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        if split == 1:
            sub_model = torch.nn.Sequential(*list(model.children())[:4])
            resources.append(sub_model)

        elif split ==2:
            sub_model = torch.nn.Sequential(*list(model.children())[4:7])
            resources.append(sub_model)

        elif split == 3:
            sub_model = torch.nn.Sequential(*list(model.children())[7:9])
            last_layer = torch.nn.Sequential(*list(model.children())[9:])
            resources.append([sub_model, last_layer])

        if len(resource_semaphores) < 3:
            resource_semaphores.append([split, 0])

    #try to send updated state to master
    heartbeat_semaphores[1] = 1


#personal nodes resource will be element 0 in resources
#model split = 0 if full model, = 1 for split 1, = 2 for split 2, and so on
def personal_execution_thread(features, labels, task_num, resources_index=0, model_split=0):
    global resources
    global resource_semaphores
    global heartbeat_semaphores
    global completed_tasks
    global is_master

    global split2_tasks
    global split3_tasks
    global return_tasks

    if model_split != 3:
        output = resources[resources_index](features) 
    else:
        output = resources[resources_index][0](features)
        output = torch.flatten(output, 1)
        output = resources[resources_index][1](features)

    if is_master:
        if (torch.argmax(output, axis=1)==labels):
            completed_tasks[1] += 1
        completed_tasks[0] += 1
        print('Task: ', task_num, ' complete on personal exec')
    else:
        if model_split == 1:
            split2_tasks.append([output, labels, task_num])
        elif model_split == 2:
            split3_tasks.append([output, labels, task_num])
        elif model_split == 3:
            output = torch.argmax(output, axis=1)
            return_tasks.append([output, task_num])
    
    resource_semaphores[resources_index] = 0

    #try to send updated state to master
    heartbeat_semaphores[1] = 1

'''
                                           -----------------------------
-------------------------------------------| scheduler thread function |-------------------------------------------
                                           -----------------------------
'''

#thread function for master node only
def scheduler_thread(sp):
    global all_task_completed
    global node_resource_semaphores
    global node_channel_semaphores
    global sp_addr_list
    global sp_connection_semaphores
    global heartbeat_messages

    global split2_tasks
    global split3_tasks 
    global return_tasks 

    #main scheduler loop
    while not all_task_completed:
        
        if len(node_resource_semaphores) > 0:
            #if  we have tasks waiting to be offloaded check through node resources
            for index, node in enumerate(node_resource_semaphores):
                for slot in node:
                    #check if avaible resource for raw image
                    if len(tasks_to_offload) > 0 and slot[0] == 1:
                        #check channel states for this node
                        channel_is_available = False
                        channel_save_index = 1
                        for i, flag in enumerate(sp_connection_semaphores[index]):
                            if flag == 0 and i != 0 and not channel_is_available:
                                sp_connection_semaphores[index][i] = 1
                                channel_is_available = True
                                channel_save_index = i

                        if channel_is_available:
                            task = tasks_to_offload.pop(0)
                            message = [5,[0,channel_save_index], [index, channel_save_index], task, 0]
                            while heartbeat_messages[index][0] != 0:
                                time.sleep(0.001)
                            heartbeat_messages[index] = message
                    
                    #check if avaible resouce for second split tasks
                    if slot[0] == 2 and len(node_resource_semaphores) >= 2:
                        sent_one_channel = False
                        cp_or_sp = 0
                        cp_or_sp_index = 0
                        channel_save_index = 1
                        receiver_index = 0
                        for i, tasks in enumerate(split2_tasks):
                            if tasks < split2_tasks[index] and i != index and not sent_one_channel:
                                #for now i need to hardcode for 3 nodes
                                channel = None
                                receiver_index = i
                                if index == 0:
                                    channel = node_channel_semaphores[index][1][i-1]
                                    cp_or_sp = 1
                                    cp_or_sp_index = i-1
                                elif index == 1:
                                    cp_or_sp_index = 0
                                    if i < 1:
                                        channel = node_channel_semaphores[index][0][0]
                                        cp_or_sp = 0
                                    else:
                                        channel = node_channel_semaphores[index][1][0]
                                        cp_or_sp = 1
                                elif index == 2:
                                    channel = node_channel_semaphores[index][0][i]
                                    cp_or_sp = 0
                                    cp_or_sp_index = i

                                for i, chan in enumerate(channel):
                                    if chan == 0 and not sent_one_channel:
                                        channel_save_index = i
                                        sent_one_channel = True

                        if sent_one_channel:
                            if index == 0:
                                receiver = 0
                            elif index == 1:
                                receiver = 0 if receiver_index == 0 else 1
                            elif index == 2:
                                receiver = 1

                            message = [4, 1, 0, (1 if cp_or_sp == 0 else 0), (receiver, channel_save_index)]
                            while heartbeat_messages[receiver_index][0] != 0:
                                time.sleep(0.001)
                            heartbeat_messages[receiver_index] = message

                            message = [4, 0, 0, cp_or_sp, (cp_or_sp_index, channel_save_index)]
                            while heartbeat_messages[index][0] != 0:
                                time.sleep(0.001)
                            heartbeat_messages[index] = message

                    #check if avaible resource for third split tasks
                    if slot[0] == 3 and len(node_resource_semaphores) > 2:
                        #hard coding for 3 nodes ONLY (see above server function for reasoning)

                        channel_save_index = 0
                        sent_one_channel = False

                        if index == 0 and split3_tasks[index] > split3_tasks[2]:

                            for i, chan in enumerate(node_channel_semaphores[index][1][1]):
                                if chan == 0 and not sent_one_channel:
                                    channel_save_index = i
                                    sent_one_channel = True

                            if sent_one_channel:
                                message = [4, 1, 1, 0, (0, channel_save_index)]
                                while heartbeat_messages[2][0] != 0:
                                    time.sleep(0.001)
                                heartbeat_messages[2] = message

                                message = [4, 0, 1, 1, (1, channel_save_index)]
                                while heartbeat_messages[index][0] != 0:
                                    time.sleep(0.001)
                                heartbeat_messages[index] = message

                        elif index == 2 and split3_tasks[index] > split3_tasks[0]:

                            for i, chan in enumerate(node_channel_semaphores[index][0][0]):
                                if chan == 0 and not sent_one_channel:
                                    channel_save_index = i
                                    sent_one_channel = True

                            if sent_one_channel:
                                message = [4, 1, 1, 1, (1, channel_save_index)]
                                while heartbeat_messages[0][0] != 0:
                                    time.sleep(0.001)
                                heartbeat_messages[0] = message

                                message = [4, 0, 1, 0, (0, channel_save_index)]
                                while heartbeat_messages[index][0] != 0:
                                    time.sleep(0.001)
                                heartbeat_messages[index] = message


        time.sleep(0.01)

'''
                                           -----------------------------
-------------------------------------------| heartbeat thread function |-------------------------------------------
                                           -----------------------------
'''

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

    global resource_semaphores
    global node_channel_semaphores
    global node_resource_semaphores

    global splits
    global context_switch_semaphore

    global split2_tasks
    global split3_tasks
    global return_tasks 

    #master to node
    # 0 basic connection do nothing
    # 1 master will be sending data of other nodes for connection to new node, ready any master port
    # 2 master telling node to ready itself for new node connection
    # 3 master telling node to context switch
    # 4 master is directing traffic, ready first master port for infromation (data for split 2 and 3)
    # 5 master will be sending raw image data to node, ready any master port (data for split 1)
    #   1, 3, and 4 will be followed by a connection port(s) in which to send data
    # 255 master is terminating connection (have node kill itself)

    #node to master
    # 0 basic connection do nothing
    # 6 node wants to send completion data to master
    # 7 node finished a task and wants to send state to master, request for port (output for split 3)
    # 255 confirmation on termination (will kill itself)

    terminate = False

#========================================================== Worker node ===========================================================

    if isinstance(protocol, ClientProtocol):
        while not terminate:
            recived = protocol.heartbeat(heartbeat_messages[0][0], cp_list[protocol_index][0])
            #for 1-4 we will need to send another heartbeat back as confirmation of request

#----------------------------------------------------------network updates----------------------------------------------------------
            if recived == 1:
                heart_check = protocol.heartbeat(1, cp_list[protocol_index][0])
                if heart_check == 1:
                    #set splits variables
                    protocol.handle_data(cp_list[protocol_index][0])
                    splits = protocol.sockets_data[protocol_index]
                    context_switch_semaphore =1 

                    #recive list of addrs to connect too
                    protocol.handle_data(cp_list[protocol_index][0])
                    addrs = protocol.sockets_data[protocol_index]

                    if isinstance(addrs, list):
                        #connect to them
                        for addr in addrs:
                            threading.Thread(target=client_connection_thread, args=([protocol], addr[0], 5555, False)).start()
                            time.sleep(0.01)

                    #return a state list of resource flags, cp flags (without master), and sp flags, and task queue lengths

                    tasks2 = len(split2_tasks)
                    tasks3 = len(split3_tasks)
                    data = [resource_semaphores, cp_connection_semaphores[1:], sp_connection_semaphores, tasks2, tasks3]

                    protocol.send_data(data, cp_list[protocol_index][0])
                    heartbeat_messages[0][0] = 0
                    heartbeat_semaphores[1] = 0

                    #try to send updated state to master
                    heartbeat_semaphores[1] = 1

            if recived == 2:
                heart_check = protocol.heartbeat(2, cp_list[protocol_index][0])
                if heart_check == 2:
                    #listen for a new connection to be made
                    threading.Thread(target=server_connection_thread, args=([secondary_protocol])).start()

                    #try to send updated state to master
                    heartbeat_semaphores[1] = 1

            if recived == 3:
                heart_check = protocol.heartbeat(3, cp_list[protocol_index][0])
                if heart_check == 3:
                    #set splits funtion
                    protocol.handle_data(cp_list[protocol_index][0])
                    splits = protocol.sockets_data[protocol_index]
                    context_switch_semaphore =1

                    #try to send updated state to master
                    heartbeat_semaphores[1] = 1 

#------------------------------------------------------intermediate data transfer-----------------------------------------------------
            if recived == 4:
                heart_check = protocol.heartbeat(4, cp_list[protocol_index][0])
                if heart_check == 4:
                    #revice state info
                    protocol.handle_data(cp_list[protocol_index][0])
                    data=protocol.sockets_data[protocol_index]

                    #sending
                    if data[0] == 0: 
                        data_to_send = split2_tasks.pop() if data[1] == 0 else split3_tasks.pop()
                        threading.Thread(target=send_message_thread, args=(protocol if data[2] == 0 else secondary_protocol, 
                                                                                data_to_send, data[3][0], data[3][1]))
                    #reciving
                    elif data[0] == 1: 
                        threading.Thread(target=receive_message_thread, args=(protocol if data[2] == 0 else secondary_protocol, 
                                                                                data[3][0], data[3][1], data[1]+1)).start()

                    #try to send updated state to master
                    heartbeat_semaphores[1] = 1

            if recived == 5:
                heart_check = protocol.heartbeat(5, cp_list[protocol_index][0])
                if heart_check == 5:
                    #recive state info
                    protocol.handle_data(cp_list[protocol_index][0])
                    addr = protocol.sockets_data[protocol_index]

                    #recv raw_image
                    cp_connection_semaphores[addr[0]][addr[1]] = 1
                    threading.Thread(target=receive_message_thread, args=(protocol, addr[0], addr[1])).start()
                
                    #try to send updated state to master
                    heartbeat_semaphores[1] = 1

#--------------------------------------------------output transfer and state updates---------------------------------------------------
            if recived == 6:
                #since we will return the argmaxed result message time is negligible
                return_data = return_tasks.pop(0)

                #check if more return data otherwise we can pass idle heartbeat
                if len(return_data) == 0:
                    heartbeat_semaphores[0] == 0

                protocol.send_data(return_data, cp_list[protocol_index][0])
                heartbeat_messages[0][0] = 0

            if recived == 7:
                #return a state list of resource flags, cp flags (without master), and sp flags, and task queue lengths

                tasks2 = len(split2_tasks)
                tasks3 = len(split3_tasks)
                data = [resource_semaphores, cp_connection_semaphores[1:], sp_connection_semaphores, tasks2, tasks3]

                protocol.send_data(data, cp_list[protocol_index][0])
                heartbeat_messages[0][0] = 0
                heartbeat_semaphores[1] = 0
                
            if recived == 255:
                terminate = True
                heartbeat_messages[0][0] = 0

            if heartbeat_semaphores[0] == 1:
                heartbeat_messages[0][0] = 6
            elif heartbeat_semaphores[1] == 1:
                heartbeat_messages[0][0] = 7

            time.sleep(0.001)

#========================================================== Master node ===========================================================
    elif isinstance(protocol, ServerProtocol):
        is_set = False
        while not all_task_completed:
            recived = protocol.heartbeat(heartbeat_messages[protocol_index][0], sp_addr_list[protocol_index][0])

            if heartbeat_messages[protocol_index][0] == 255:
                terminate = True

#--------------------------------------------------output transfer and state updates---------------------------------------------------

            #we will only care about node requests if master node is at idle with the node
            #we will need to send a heartbeat back as confirmation of request
            if heartbeat_messages[protocol_index][0] == 0:
                if recived == 6:
                    heart_check = protocol.heartbeat(6, sp_addr_list[protocol_index][0])
                    if heart_check == 6:
                        #recive task completion data
                        protocol.handle_data(sp_addr_list[protocol_index][0])
                        idx = protocol.addresses.index(sp_addr_list[protocol_index][0])
                        state_data = protocol.datas[idx]

                        #add to stats
                        completed_tasks[0] += 1
                        if data == labels:
                            completed_tasks[1] += 1
                        print('Task: ', task_num, ' complete on offloaded')

                if recived == 7 and is_set:
                    heart_check = protocol.heartbeat(7, sp_addr_list[protocol_index][0])
                    if heart_check == 7:
                        #recive state data
                        protocol.handle_data(sp_addr_list[protocol_index][0])
                        idx = protocol.addresses.index(sp_addr_list[protocol_index][0])
                        state_data = protocol.datas[idx]

                        #update resource states
                        for i, data in enumerate(state_data[0]):
                            node_resource_semaphores[protocol_index][i] = data

                        #update channel states
                        node_channel_semaphores[protocol_index] = state_data[1:3]

                        #update recoreded node queues
                        split2_tasks[protocol_index] = state_data[3]
                        split3_tasks[protocol_index] = state_data[4]

#----------------------------------------------------------network updates----------------------------------------------------------
            if recived == 1:
                connection_list = []
                splits = []

                #collect splits data
                for split in node_resource_semaphores[protocol_index]:
                    splits.append(split[0])

                #collect address
                for i, l in enumerate(sp_addr_list):
                    if i != len(sp_addr_list) - 1:
                        connection_list.append(l[0][0])
                        while heartbeat_messages[i][0] != 0:
                            time.sleep(0.001)
                        heartbeat_messages[i][0] = 2
                    
                time.sleep(0.01) #wait 10ms for nodes to ready themselves

                #send split data first then addresses
                protocol.send_data(splits, sp_addr_list[protocol_index][0])
                protocol.send_data(connection_list, sp_addr_list[protocol_index][0])

                #recive state data
                protocol.handle_data(sp_addr_list[protocol_index][0])
                idx = protocol.addresses.index(sp_addr_list[protocol_index][0])
                state_data = protocol.datas[idx]

                #update resource states
                for i, data in enumerate(state_data[0]):
                    node_resource_semaphores[protocol_index][i] = data

                #update channel states
                node_channel_semaphores[protocol_index].append(state_data[1:3])

                #update recoreded node queues
                split2_tasks[protocol_index] = state_data[3]
                split3_tasks[protocol_index] = state_data[4]


                is_set = True
                heartbeat_messages[protocol_index][0] = 0

            if recived == 2:
                heartbeat_messages[protocol_index][0] = 0

            if recived == 3:
                splits = []
                #collect splits data
                for split in node_resource_semaphores[protocol_index]:
                    splits.append(split[0])

                #send split data
                protocol.send_data(splits, sp_addr_list[protocol_index][0])

#------------------------------------------------------intermediate data transfer-----------------------------------------------------
            if recived == 4:
                data = heartbeat_messages[protocol_index][1:]
                protocol.send_data(data, sp_addr_list[protocol_index][0])
                heartbeat_messages[protocol_index] = [0,0,0,0,0]

            if recived == 5:
                data = heartbeat_messages[protocol_index][1]
                print("sending data??")
                protocol.send_data(data, sp_addr_list[protocol_index][0])
                
                threading.Thread(target=send_message_thread, args=(protocol, heartbeat_messages[protocol_index][3], 
                                                                             heartbeat_messages[protocol_index][2][0], 
                                                                             heartbeat_messages[protocol_index][2][1])).start()

                heartbeat_messages[protocol_index] = [0,0,0,0,0]
            
            time.sleep(0.001)

'''
                                           ----------------------------
-------------------------------------------| message thread functions |-------------------------------------------
                                           ----------------------------
'''

#this is only for sending data over channel                
def send_message_thread(protocol, data, protocol_index, channel_selection=0):
    global cp_list
    global cp_connection_semaphores
    global sp_addr_list
    global sp_connection_semaphores
    global is_master

    if isinstance(protocol, ClientProtocol):
        protocol.send_data(data, cp_list[protocol_index][channel_selection])

        #this should be set to 1 before calling this thread function
        cp_connection_semaphores[protocol_index][channel_selection] = 0

    elif isinstance(protocol, ServerProtocol):
        protocol.send_data(data, sp_addr_list[protocol_index][channel_selection])

        #this should be set to 1 before calling this thread function
        sp_connection_semaphores[protocol_index][channel_selection] = 0

    if not is_master:
        #try to send updated state to master
        heartbeat_semaphores[1] = 1

#this is only for receiving data over channel  
def receive_message_thread(protocol, protocol_index, channel_selection=0, data_type=0):
    global cp_list
    global cp_connection_semaphores
    global sp_addr_list
    global sp_connection_semaphores
    global is_master
    
    global raw_img_tasks 
    global split2_tasks 
    global split3_tasks 

    if isinstance(protocol, ClientProtocol):
        protocol.handle_data(cp_list[protocol_index][channel_selection])
        data = protocol.sockets_data[cp_list[protocol_index][channel_selection]]

        if data_type == 0:
            raw_img_tasks.append(data)
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

        if data_type == 1:
            split2_tasks.append(data)
        elif data_type == 2:
            split3_tasks.append(data)
        elif data_type == 3:
            completed_tasks[0] += 1
            if data == labels:
                completed_tasks[1] += 1

        #this should be set to 1 before calling this thread function
        sp_connection_semaphores[protocol_index][channel_selection] = 0

    if not is_master:
        #try to send updated state to master
        heartbeat_semaphores[1] = 1

'''
                                                    ----------------
----------------------------------------------------| main thread  |----------------------------------------------------
                                                    ----------------
'''

if __name__ == '__main__':

#========================================================== Master node ===========================================================

    if sys.argv[1] == '--master': 
        print('starts as master')

        #set inital variables
        is_master = True
        thread_list = []
        master_ip = sys.argv[2][2:]
        master_port = int(sys.argv[3][2:])

        #spin up server protocol
        sp = ServerProtocol(is_master)
        sp.listen(master_ip, master_port)
        thread_list.append(threading.Thread(target=server_connection_thread, args=([sp])).start())

        #set resources/load model and dataset

        resources.append(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True))
        resource_semaphores = [0]

        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((500,500)),
                                                torchvision.transforms.ToTensor()])
        imagenet_data = torchvision.datasets.ImageNet('J:\\ImageNet\\Val1per', transform = transform, split='val')
        data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=False) 

        #task generation loop
        frame_rate = 0.0334 #simulated frame rate timing (0.0334 ~ 29.97 fps)

        personal_task_start_time = 0
        task_num = 0 #also iteration count
        personal_task_processing_number = 0
        last_personal_task_process = 1 #begining of last batch of personal tasks

        current_num_tasks_offloaded = 0

        #Spin up schdeuler thread 
        thread_list.append(threading.Thread(target=scheduler_thread, args=([sp])).start())

        print("Waiting 10 Seconds to allow connections")
        time.sleep(10)

        with torch.no_grad():
            for features, labels in data_loader:
                frame_capture_start = time.time()

                task_num += 1
                raw_img_tasks.append([features, labels, task_num])

                #if personal resource is avaible send oldest frame saved to personal resource 
                if resource_semaphores[0] == 0:  
                    task = raw_img_tasks.pop(0)
                    personal_task_processing_number = task[0]

                    personal_task_start_time = time.time()
                    threading.Thread(target=personal_execution_thread, args=(task[0], task[1], task[2])).start()

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

            while len(raw_img_tasks) > 0:
                #check if personal resource is avaible send oldest frame saved to personal resource 
                if resource_semaphores[0] == 0:  
                    task = raw_img_tasks.pop(0)
                    personal_task_processing_number = task[0]

                    personal_task_start_time = time.time()
                    threading.Thread(target=personal_execution_thread, args=(task[0], task[1], task[2])).start()

                    resource_semaphores[0] = 1
                time.sleep(0.01)

        for thread in thread_list:
            if thread != None:
                thread.join()
        
        all_task_completed = True

#========================================================== Worker node ===========================================================

    else:
        print('starts as worker')
        
        thread_list = []
        master_ip = sys.argv[2][2:]
        master_port = int(sys.argv[3][2:])

        cp = ClientProtocol()
        sp = ServerProtocol(is_master)

        sp.listen(master_ip, master_port+1)

        #begin connection
        thread_list.append(threading.Thread(target=client_connection_thread, args=(cp, master_ip, master_port, True, sp)).start())

        #wait for connections and threads to set then establish splits
        time.sleep(3)
        set_model_splits(splits)

        #wait until resources are set (can possibly remove this but just to be safe for now)
        while len(resources) < 3:
            time.sleep(0.01) 

        while not all_task_completed:
            #if a context switch need to be preformed do that by call set_model_splits again
            if context_switch_semaphore == 1:
                set_model_splits(splits)

            #loop through resource semaphore to check for tasks
            for index, resource in enumerate(resource_semaphores):
                print(resource, raw_img_tasks)
                if resource[0] == 1 and resource[1] == 0 and len(raw_img_tasks) > 0:
                    resource_semaphores[index][1] = 1
                    #try to send updated state to master
                    heartbeat_semaphores[1] = 1

                    task = raw_img_tasks.pop(0)
                    threading.Thread(target=personal_execution_thread, args=(task[0], task[1], task[2], index, resource[0])).start()

                elif resource[0] == 2 and resource[1] == 0 and len(split2_tasks) > 0: 
                    resource_semaphores[index][1] = 1
                    #try to send updated state to master
                    heartbeat_semaphores[1] = 1

                    task = split2_tasks.pop(0)
                    threading.Thread(target=personal_execution_thread, args=(task[0], task[1], task[2], index, resource[0])).start()

                elif resource[0] == 3 and resource[1] == 0 and len(split3_tasks) > 0:
                    resource_semaphores[index][1] = 1
                    #try to send updated state to master
                    heartbeat_semaphores[1] = 1

                    task = split3_tasks.pop(0)
                    threading.Thread(target=personal_execution_thread, args=(task[0], task[1], task[2], index, resource[0])).start()
            
            #small delay to prevent over consumption of machines resources
            time.sleep(0.0001)

        for thread in thread_list:
            if thread != None:
                thread.join()
        
        