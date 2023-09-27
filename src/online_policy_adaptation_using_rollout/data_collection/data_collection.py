import random
import numpy as np
import joblib
import paramiko
import subprocess
import time
import requests
import sys
import os
from pathlib import Path

'''
This function applies the configuration on the master node. This configuration includes routing weights, scalings and blocking upper limits
'''

# x.x.x.x:xxxx (IP and port of book and city services services on the cluster) /home/shahab/searchapp3/data_collection_for_dropping/ (path to the command to file and routing config file on the master node) username (of your account on the master node) password (of your account on the master node) x.x.x.x (IP of master node)

args = sys.argv
IP_port = args[1]
path_to_routing_config = args[2]
path_to_scaling_config = args[3]
server_username = args[4]
server_password = args[5]
server_hostname = args[6]

artificas = "./artifacts/"
path_to_LG = ""


def take_routing_action_master_node(p11, p21, path_to_routing_config, server_hostname, server_username,
                                    server_password):
    '''
    :param p11: The routing weight of service 1 towards the computeinfo node 1.
    The remaining traffic will route towards node 2
    :param p21: The routing weight of service 2 towards the computeinfo node 1.
    The remaining traffic will route towards node 2
    :param path_to_routing_config: This is the path to the file that convert the command to configuration file on the master node.
    :param server_hostname: This is the ip address of the master node.
    :param server_username: This is the username defined on the master node.
    :param server_password: This is the password defined on the master node.
    :return: None

    In this funtion, we change the routing configration file for the services towards the node1 and node2 and then apply
    the configuration file on the master node.
    '''
    routing_weight1 = p11 * 100
    routing_weight2 = p21 * 100
    #################Taking routing actions################
    hostname = server_hostname
    username = server_username
    password = server_password
    commands = ["python " + path_to_routing_config + "/command_to_file.py " +
                str(int(routing_weight1)) + " " + str(int(routing_weight2))]
    commands.append("kubectl apply -f " + path_to_routing_config + "/user_based_weight.yaml")
    client = paramiko.SSHClient()
    # add to known hosts
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=hostname, username=username, password=password)
    except:
        print("[!] Cannot connect to the SSH Server")
        exit()

    for command in commands:
        print("=" * 50, command, "=" * 50)
        stdin, stdout, stderr = client.exec_command(command)
        err = stderr.read().decode()
        if err:
            print("The error is:" + err)
    client.close()
    pass

def take_blocking_action_master_node(IP_port):
    '''
    :param IP_port: This is the ip and port address of the services running on the cluster.
    Please note that the ip and port are the same for both services but the URLs are different.
    :return:

    In this function we specify the upper bound of the number of requests that will be replied.
    If the rate is higher than the expetation, the front node will reply by the response with the code 429.
    '''
    #################Taking blocking actions###############
    r = requests.get('http://'+IP_port+'/ratebook/' + str(limit1))
    print(r.text)
    r = requests.get('http://'+IP_port+'/ratecity/' + str(limit2))
    print(r.text)


def take_blocking_action_master_node(limit1, limit2, IP_port, path_to_routing_config, server_hostname, server_username,
                                     server_password):
    r = requests.get('http://' + IP_port + '/rateinfo/' + str(limit2))
    print(r.text)
    r = requests.get('http://' + IP_port + '/ratecompute/' + str(limit1))
    print(r.text)
    pass


def take_scaling_action_on_master_node(c1, c2, IP_port, path_to_scaling_config, server_hostname, server_username,
                                       server_password):
    '''
    :param c1: This is the scaling action on node 1
    :param c2: This is the scaling action on node 2
    :param IP_port: this is the ip and port of the serviecs on the cluster
    :param path_to_scaling_config: This is the path to the file to convert the scaling action to the configuration file on the master node.
    :param server_hostname: This is the ip address of the master node.
    :param server_username: The username defined on the master node.
    :param server_password: The password for the above-mentioed username on the master node.
    :return: None

    In this function, we change the allocated CPU to the computeinfo nodes and apply the configuration on the master node.
    '''
    #################Taking scaling actions################
    scaling1 = c1 * 1000
    scaling2 = c2 * 1000
    #################Taking routing actions################
    hostname = server_hostname
    username = server_username
    password = server_password
    commands = ["python " + path_to_scaling_config + "/command_to_file.py 1 " + str(int(scaling1)) + " " + str(
        int(scaling1)) + " computeinfo1.yaml"]
    commands.append("kubectl apply -f " + path_to_scaling_config + "/computeinfo1.yaml")
    commands.append("python "+path_to_scaling_config+"/command_to_file.py 1 " + str(int(scaling2)) + " " + str(int(scaling2)) + " computeinfo2.yaml")
    commands.append("kubectl apply -f "+path_to_scaling_config+"/computeinfo2.yaml")
    client = paramiko.SSHClient()
    # add to known hosts
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=hostname, username=username, password=password)
    except:
        print("[!] Cannot connect to the SSH Server")
        exit()

    for command in commands:
        print("=" * 50, command, "=" * 50)
        stdin, stdout, stderr = client.exec_command(command)
        err = stderr.read().decode()
        if err:
            print("The error is:" + err)
    client.close()


def run_load_generator(LG_name, load_value):
    '''
    :param LG_name: This paramter specifes the name of the service that we like to load
    :param load_value: Load value is the number requests per second (
    Plese note that this is when we like to specify the constant load for the specific time interval.
    We can also specify the load pattern in a file and call the load generator with those parameters.)
    :return: None

    This function start the load generator for the specific service with the specific load pattern.
    In the case of data collection we call the load generator with the contant load for a specif time interval in which the system has specific configuration.
    '''
    # Define the shell command to run
    shell_cmd = "nohup python " + path_to_LG + "async_load_generator.py " + LG_name + " constant " + str(
        load_value) + " " + IP_port + " > /dev/null &"

    # Run the command in the background
    subprocess.Popen(shell_cmd, shell=True)


def kill_load_generator():
    '''
    This function is used to kill all the load generators before or after the process of load generarion.
    :return: None
    '''
    # Define the shell command to run
    shell_cmd = "pkill -f async_load_generator.py"

    # Run the command in the background
    subprocess.Popen(shell_cmd, shell=True)


'''
The load pattern states the values of load that should be generated over time. For instance, the load pattern with values of [5,5,10,15,20] means that over the first time step (in this work is 5 sec) is 5, the second time step it is 5, the third is 15 and so on. 

This function accepts the name of file which includes the load pattern and it returns the load pattern as a set
'''


def get_load_pattern(file_name):
    '''
    We use this function to read specific load pattern from a file. In the case of data collection, we do not use this function.
    :param file_name: Name of the file that includes the load pattern
    :return: This file returns the set consisting of load values.
    '''
    with open(file_name) as file:
        lines = file.readlines
    file.close()
    load_pattern = []
    for line in lines:
        load_pattern.append(float(line))
    return load_pattern


def aggregate_samples(number_of_samples, p11, p21, c1, path_to_files, expected_l1, expected_l2, Timestep=5):
    '''
    We use this function to collect samples for specific load value and specific configuration within specified time steps.

    :param number_of_samples: This parameter specifies how many samples we like to collect for the given load value and configuration.
    We usually need to collect more than one sample on a real testbed sine the system is not deterministic.
    :param p11: The weight of service 1 towards node 1
    :param p21: The weight of service 1 towards node 2
    :param c1: The scaling action for node 1
    :param c2: The scaling action for node 2
    :param path_to_files: The path to the file that we save the samples.
    :param expected_l1: This is the load that we send towards the nodes.
    However, in a real system the load that is sent may be slightly larger or smaller than this value.
    :param expected_l2: This is the load that we send towards the nodes.
    However, in a real system the load that is sent may be slightly larger or smaller than this value.
    :param Timestep: this is the time step of our system when we model it as a discrete model
    :return: None
    '''

    counter = 0

    while counter < number_of_samples:
        print("The counter is:", counter)
        ##############First we clean up all the files
        with open(path_to_files + "load_compute.txt", 'w') as f:
            f.write("")
        f.close()
        with open(path_to_files + "response_compute.txt", "w") as f:
            f.write("")
        f.close()

        with open(path_to_files + "load_info.txt", 'w') as f:
            f.write("")
        f.close()
        with open(path_to_files + "response_info.txt", "w") as f:
            f.write("")
        f.close()

        ###############Wait for the time step
        time.sleep(Timestep)
        ###############Read the state
        with open(path_to_files + "load_compute.txt", 'r') as f:
            lines = f.readlines()
        f.close()

        offered_load = len(lines)

        with open(path_to_files + "response_compute.txt", "r") as f:
            lines = f.readlines()
        f.close()

        info_res_1 = []
        info_res_2 = []
        info_drops = []
        for line in lines:
            req = eval(line)
            if (req["code"] == 200):
                if (req["Node"] == 1):
                    info_res_1.append(req["response"])
                elif (req["Node"] == 2):
                    info_res_2.append(req["response"])
            else:
                info_drops.append(req["response"])
        ###########Info###############
        info_all_req = info_res_1 + info_res_2
        l1 = offered_load
        cl1 = len(info_res_1) + len(info_res_2)
        if (len(info_res_1) > 0):
            d11 = np.mean(info_res_1)
            d11_std = np.std(info_res_1)
        else:
            d11 = 0
            d11_std = 0

        if (len(info_res_2) > 0):
            d12 = np.mean(info_res_2)
            d12_std = np.std(info_res_2)
        else:
            d12 = 0
            d12_std = 0

        if (len(info_all_req) > 0):
            d1 = np.mean(info_all_req)
            d1_99 = np.percentile(info_all_req, 99)
            d1_90 = np.percentile(info_all_req, 90)
            d1_std = np.std(info_all_req)
        else:
            d1 = 0
            d1_99 = 0
            d1_90 = 0
            d1_std = 0

        drop1 = len(info_drops)

        ########Read for service compute#########
        with open(path_to_files + "load_compute.txt", 'r') as f:
            lines = f.readlines()
        f.close()

        offered_load = len(lines)

        with open(path_to_files + "response_compute.txt", "r") as f:
            lines = f.readlines()
        f.close()

        compute_res_1 = []
        compute_res_2 = []
        compute_drops = []
        for line in lines:
            req = eval(line)
            if (req["code"] == 200):
                if (req["Node"] == 1):
                    compute_res_1.append(req["response"])
                elif (req["Node"] == 2):
                    compute_res_2.append(req["response"])
            else:
                compute_drops.append(req["response"])
        ###########Book###############
        compute_all_req = compute_res_1 + compute_res_2
        l2 = offered_load
        cl2 = len(compute_res_1) + len(compute_res_2)
        if (len(compute_res_1) > 0):
            d21 = np.mean(compute_res_1)
            d21_std = np.std(compute_res_1)
        else:
            d21 = 0
            d21_std = 0

        if (len(compute_res_2) > 0):
            d22 = np.mean(compute_res_2)
            d22_std = np.std(compute_res_2)
        else:
            d22 = 0
            d22_std = 0

        if (len(compute_all_req) > 0):
            d2 = np.mean(compute_all_req)
            d2_99 = np.percentile(compute_all_req, 99)
            d2_90 = np.percentile(compute_all_req, 90)
            d2_std = np.std(compute_all_req)
        else:
            d2 = 0
            d2_99 = 0
            d2_90 = 0
            d2_std = 0

        drop2 = len(compute_drops)

        ##############Input of the system model################
        # l1 is the offered load that LG sends out
        # p11 is the routing weight of service 1 toward node 1, p21 is the routing weight of service 2 toward node 1
        # p12 = 1 - p11 that is why we dont save it here.
        # b1 and b2 the blocking rates.
        # l1,l2,p11,p21,b1,b2

        ##############Out put of system model#########
        # cl1, cl2 are the carried load of the system. Carried load is the part of load that system will respond and not drops
        # d1, d2 are the average response time of service 1 and 2 over the specified time step which is 5 sec here.
        # d1_std and d2_std are standard deviation
        # d11, d12,d21,d22 these are the average response time for service i on node j
        # d11_std, d12_std, d21_std, d22_std which are standard deviations
        # drop 1, drop 2 which are the number of dropped requests.
        # cl1, cl2, d1, d2,d1_std,d2_std, d11, d12, d21, d22, d11_std, d12_std, d21_std, d22_std, drop1, drop2

        line = (str(expected_l1) + "," + str(expected_l2) + "," + str(l1) + "," + str(l2) + "," + str(b1) + "," + str(b2)
                + "," + str(p11) + "," + str(p21) + "," + str(c1) + "," + str(c2) + "," + str(cl1)+ "," + str(cl2) + ","
                + str(d1) + "," + str(d1_std) + "," + str(d11) + "," + str(d12) + "," + str(d11_std) + "," + str(d12_std)
                + "," + str(drop1) + "," + str(d1_99) + "," + str(d1_90)+ str(d2) + "," + str(d2_std) + "," + str(d21)
                + "," + str(d22) + "," + str(d21_std) + "," + str(d22_std)
                + "," + str(drop2) + "," + str(d2_99) + "," + str(d2_90) + "\n")

        with open(path_to_files + "data.csv", 'a') as f:
            f.write(line)
        f.close()
        counter += 1


def cleanup_files():
    '''
    We use this function to clean up all related files from previous runs.
    :return: None.
    '''
    file.close()
    path = Path(artificas + "load_compute.txt")
    if (path.is_file()):
        os.remove(artificas + "load_compute.txt")
    path = Path(artificas + "response_compute.txt")
    if (path.is_file()):
        os.remove(artificas + "response_compute.txt")


if __name__ == "__main__":
    # load1 = get_load_pattern("load_pattern_for_service_1")
    # load2 = get_load_pattern("load_pattern_for_service_2")
    info_load = [5, 10, 15, 20]
    compute_load = [1, 2, 3, 4, 5]
    actions = [0, 0.2, 0.4, 0.6, 0.8, 1]
    scaling_actions = [1, 2, 3, 4, 5]

    kill_load_generator()

    path = Path(artificas + "data.csv")
    if (path.is_file()):
        os.remove(artificas + "data.csv")
    with open(artificas + "data.csv", 'w') as file:
        file.write("expected_l1,expected_l2,l1,l2,b1,b2,p11,p21,c1,c2,cl1,cl2,d1,d1_std,d11,d12,d11_std,d12_st,drop1,d1_99,d1_90,"
                   "d2,d2_std,d21,d22,d21_std,d22_st,drop2,d2_99,d2_90\n")

    time_step = 5

    for l1 in info_load:
        for l2 in info_load:
            run_load_generator("compute",l2)
            run_load_generator("info",l1)
            time.sleep(30)
            cleanup_files()
            for p11 in actions:
                for p21 in actions:
                    for b1 in actions:
                        for b2 in actions:
                            limit1 = (1-b1) * l1
                            limit2 = (1-b2) * l2
                            take_routing_action_master_node(p11,p21, IP_port, path_to_routing_config, server_hostname, server_username, server_password)
                            take_blocking_action_master_node(limit1,limit2, IP_port, path_to_routing_config, server_hostname, server_username, server_password)
                            aggregate_samples(2,p11, p21, b1, b2,artificas,l1,l2,time_step)
            kill_load_generator()