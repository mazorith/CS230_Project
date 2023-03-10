import torch
import torchvision
import thread
import sys
import time
import random
from queue import PriorityQueue
from PIL import Image
from MasterNode import ClientProtocol
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

#connection lists
cp_list = []
#this will be a list of lists representing the server connections per device
sp_addr_list = []
#only for master node will be a list of lists of tuples #(is_running, model_split) for each node in network
node_resource_semaphores = [] 

#lists containing the data inputs for the model
#TODO: change lists to priority queues for time squental processing 

raw_img_tasks = []
split2_tasks = []
model3_tasks = []

#TODO: add complexity to completed_tasks
completed_tasks = (0,0) #currently (# of tasks complete, # of tasks correct [top 1%])

#list of models splits for the node, will contain 1 for master and 3 for worker nodes
resources = []
resource_semaphores = [0]
#min_model_exc_time = 0.82
avg_model_exc_time = 1.08
avg_batch_model_offload_time = [2.16, 2.30, 1.87] #from calculations using recorded data - batch of [3,6,6]
avg_offload_factors = [0.76,0.73,0.36] #customizable modification factors for avg_batch_model_offload_time

def server_connection_thread(sp):
    global is_master
    global sp_addr_list
    global node_resource_semaphores

    #TODO: add functionality to start heartbeat messages??

    if is_master:
        while is_master:
            ret_list = sp.server_handshake()
            sp_addr_list.append(ret_list)

            #optimal split configurations without dynamic contex switching
            num_of_nodes = len(sp_addr_list)
            if num_of_nodes == 1:
                #default split (1,2,3)
                pass
            elif num_of_nodes == 2:
                #2 node split (1,2,2) - (2,2,3)
                pass
            elif num_of_nodes == 3:
                #default + 2 node split (1,2,3) - (1,2,2) - (2,2,3)
                pass

            node_resource_semaphores.append([(0,0),(0,0),(0,0)])

    else:
        ret_list = sp.server_handshake()
        sp_addr_list.append(ret_list)
        node_resource_semaphores.append([(0,0),(0,0),(0,0)])

#personal nodes resource will be element 0 in resources
def personal_execution(features, labels):
    global resources
    global resources_semaphores
    global completed_tasks

    resources[0](features) 

    #TODO:create code for checking accuracy

    completed_tasks[0] += 1
    resources_semaphores[0] = 0


if __name__ == '__main__':
    #TODO: refactor to have master ip as input of sys.argv[2], and port as sys.argv[3]
    if sys.argv[1] == '--master': 
        print('starts as master')

        #set inital variables
        global is_master
        is_master = True
        thread_list = []

        #spin up server protocol
        sp = ServerProtocol()
        sp.listen('127.0.0.1', 5555)
        thread_list.append(threading.Thread(target=server_connection_thread, args=(sp)))

        #set resources/load model and dataset
        global resources 
        global resources_semaphores

        resources.append(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True))
        resources_semaphores = 0

        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((500,500)),
                                                torchvision.transforms.ToTensor()])
        imagenet_data = torchvision.datasets.ImageNet('J:\\ImageNet', transform = transform, split='val')
        data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=False) 

        #task generation loop
        global sp_addr_list
        global raw_image_tasks #remove global functionality of this variable for master
        global avg_model_exc_time

        global avg_batch_model_offload_time
        global avg_offload_factors

        frame_rate = 0.0334 #simulated frame rate timing (0.0334 ~ 29.97 fps)

        personal_task_start_time = 0
        task_num = 0 #also iteration count
        personal_task_processing_number = 0
        last_personal_task_process = 1 #begining of last batch of personal tasks

        #TODO: Make this a global variable
        current_num_tasks_offloaded = 0

        #TODO: Create scheduler node thread function to do scheduling

        with torch.no_grad():
            for features, labels in data_loader:
                frame_capture_start = time.time()

                task_num += 1
                raw_image_tasks.append((task_num, features, lables))

                #if personal resource is avaible send oldest frame saved to personal resource 
                if resource_semaphores[0] == 0:  
                    task = raw_image_tasks.pop(0)
                    personal_task_processing_number = task[0]

                    personal_task_start_time = time.time()
                    threading.Thread(target=personal_execution, args=(task[1], task[2]))

                    resource_semaphores[0] = 1

                #if there are other nodes in the network and queued tasks decide if newest task should be offloaded
                if len(sp_addr_list) > 0 and len(raw_image_tasks) > 0:
                    #estimate time need to run the personal resource including current execution
                    personal_time_est = (task_num - last_personal_task_process - 1)*avg_model_exc_time
                    personal_time_est += (avg_model_exc_time - (time.time() - personal_task_start_time)) 
                    
                    #get calulated single task estimates
                    network_config = len(sp_addr_list)-1
                    offload_time_est = avg_batch_model_offload_time[network_config]

                    #depending on load of the system add additional delay(may need to remove /2 for avg_offload_factors)
                    if current_num_task_offloaded >= 3 and network_config == 0:
                        offload_time_est += (current_num_task_offloaded - 2) * (avg_offload_factors[network_config]/2)
                    elif current_num_task_offloaded >= 6 and network_config > 0:
                        offload_time_est += (current_num_task_offloaded - 5) * (avg_offload_factors[network_config]/2)
                    
                    #if offload estimate is less that personal execution estimate offload task
                    if offload_time_est <= personal_time_est:
                        #TODO: edit this offloaded var for global
                        current_num_task_offloaded += 1

                        #TODO: call thread to send task to first offload machine
                        task_to_send = raw_image_tasks.pop()

                #simulate a framerate 
                if frame_rate > time.time() - frame_capture_start:
                    time.sleep(frame_rate - (time.time() - frame_capture_start))

    else:
        print('starts as worker')

        #set inital variables
        global cp_list
        #TODO: cp_list will store connections in for of (ip, port) -- refactor to commandline args for master node
        thread_list = []

        cp = ClientProtocol()

        for i in range(4):
            cp.connect('127.0.0.1', 5555)
            cp.client_handshake(i)

        #get initial message from master node
        #if first node server protocol not needed else create server protocol
        
        #once connections are established wait for master to direct resource slots
        #simply loop heartbeat and functionality threads until termination heartbeat is sent by master 
        