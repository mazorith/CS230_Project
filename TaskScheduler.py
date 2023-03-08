import torch
import torchvision
import thread
import time
import random
from queue import PriorityQueue
from PIL import Image
from MasterNode import ClientProtocol
from ServerNode impoer SeverProtocol

#using global variables to share data between threads

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
resources = None
resource_semaphores = None
min_model_exc_time = 0.82
avg_model_exc_time = 1.08

def server_connection_thread(sp):
    global is_master
    global sp_addr_list
    global node_resource_semaphores

    if is_master:
        while is_master:
            ret_list = sp.server_handshake()
            sp_addr_list.append(ret_list)
            node_resource_semaphores.append([(0,0),(0,0),(0,0)])

    else:
        ret_list = sp.server_handshake()
        sp_addr_list.append(ret_list)
        node_resource_semaphores.append([(0,0),(0,0),(0,0)])

def personal_execution(features, labels):
    global resources
    global resources_semaphores
    global completed_tasks

    resources_semaphores = 1
    resources(features) 

    #TODO:create code for checking accuracy

    completed_tasks[0] += 1
    resources_semaphores = 0

if __name__ == '__main__':
    if sys.argv[1] == '--master':
        print('starts as master')

        is_master = True
        thread_list = []

        sp = ServerProtocol()
        sp.listen('127.0.0.1', 5555)
        thread_list.append(threading.Thread(target=server_connection_thread, args=(sp)))

        global resources 
        global resources_semaphores

        #TODO: load model into resources
        resources = None
        resources_semaphores = 0

        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((500,500)),
                                                torchvision.transforms.ToTensor()])

        imagenet_data = torchvision.datasets.ImageNet('J:\\ImageNet', transform = transform, split='val')
        data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=False) 

        start_time = 0
        with torch.no_grad():
            for features, labels in data_loader:
                raw_image_tasks.append(features, lables)
                if resource_semaphores == 0:  
                    start_time = time.time()
                    threading.Thread(target=personal_execution, args=(features, labels))

    else:
        print('starts as worker')