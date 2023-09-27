#
import asyncio
import time
import random
import aiohttp
# import pika
import numpy as np
import sys

########################## Call function with the following order of the input variables################
# python async_load_generator.py service_name type_of_load value_or_name_of_load_pattern IP_port
# python async_load_generator.py city constant 10 x.x.x.x:30037

### Initialization of the parameters
np.random.seed(0)

# The parameters of the load generator
MIN_INTERVAL = 0.05    # Minimum interval between two requests on the real system
CLIENTS_NO = 1
LD_ID = 1

path_to_artifacts = "./artifacts/"

###############################Input arguments##############################
# The arguments specify which service and which patter should be considered for load generation.
############################################################################
args = sys.argv

service_name = args[1]

load_pattern = args[2]

IP_port = args[4]

if service_name == "compute":
    url = "http://"+IP_port+"/compute"
    username = "premium"
    passwd = "1234"
    load_file_name = path_to_artifacts + "load_compute.txt"
    response_file_name = path_to_artifacts + "response_compute.txt"
    LD_ID = 2
    word1 = ", node1,"
    word2 = ", node2,"
elif service_name == "background":
    url = "http://"+IP_port+"/compute"
    username = "fremium"
    passwd = "1234"
    load_file_name = path_to_artifacts + "load_background.txt"
    response_file_name = path_to_artifacts + "response_backgroun.txt"
    LD_ID = 5
    word1 = ", node1,"
    word2 = ", node2,"
elif service_name == "info":
    url = "http://"+IP_port+"/info"
    username = "fremium"
    passwd = "1234"
    load_file_name = path_to_artifacts + "load_info.txt"
    response_file_name = path_to_artifacts + "response_info.txt"
    LD_ID = 1
    word1 = ", node1,"
    word2 = ", node2,"
elif service_name == "book":
    url = "http://"+IP_port+"/books"
    username = "premium"
    passwd = "1234"
    load_file_name = path_to_artifacts + "load_book.txt"
    response_file_name = path_to_artifacts + "response_book.txt"
    LD_ID = 3
    word1 = "(fetch3)"
    word2 = "(fetch2)"
else:
    url = "http://"+IP_port+"/city"
    username = "fremium"
    passwd = "1234"
    load_file_name = path_to_artifacts + "load_city.txt"
    response_file_name = path_to_artifacts + "response_city.txt"
    LD_ID = 3
    word1 = "(fetch3)"
    word2 = "(fetch2)"

login_url = 'http://'+IP_port+'/login'

################################Load generation methods################################
# Here we specify the function for the load pattern. For instance, the load pattern can be a constant load pattern.
# For constant load pattern the intervals are based on the number of requests that we send out per second.
# Moreover, we can read the load pattern from a file, or we can send requests based on the Poisson process modulated with specific function.
# The output of these functions is the time that sender should wait for each request from the time that the load generator is started.
# We can add a delay in the beginning of the time stamps so the load generator start sending the load after the specified delay.
# In the functions with the load specified in a file, we can see in the loop of delay generation, we have two parameters,
# step_size and iteration. step size specifies the time step in our system and iteration means how many of these time steps we like to have same load value.
# For simplification, we could have iteration equal to 1 but in that case we need to repeat the load value in the file. This function can be modified simply.

def constant(load):
    T = 7000
    delays = []
    interval = 1/load
    delay = interval
    while delay<T:
        delays.append(delay)
        delay += interval
    return delays

def from_load_pattern(file_name, step_size, iteration):
    delays = []
    loads = []
    with open(file_name) as file:
        lines = file.readlines()
    for line in lines:
        loads.append(float(line))

    delay = 0
    l = []
    for i in range(1,len(loads)+1):
        interval = 1/loads[i-1]
        print(interval)
        delay += interval
        while delay < iteration * step_size *i:
            delays.append(delay)
            delay += interval
            l.append(interval)
    return delays, l

def random_constant_load():
    delays = []
    loads = [20]
    delay = 0
    l = []
    for i in range(1,4):
        load = loads[random.randint(0,len(loads)-1)]
        interval = 1/load
        print(interval)
        delay += interval
        while delay < 100*i:
            delays.append(delay)
            delay += interval
            l.append(interval)
    return delays, l

def poisson_modulated_by_f(name):

    Time = 7200
    t = 0
    k = 0
    intervals = []

    lam0 = 30
    A = 10
    T = 1200
    S = []

    lam = lam0 + A + 1

    S.append(t)
    t_old = 0

    while (t < Time):
        r = random.random()
        t = t - np.log(r) / lam
        if (t > Time):
            break
        s = random.random()
        # The average of the Poisson  process as a function (Poisson  process modulated by function f- constant or sin)
        lamT = constant(t, T, lam0, A)
        # A * np.sin((2 * np.pi * t) / (T))
        if (s<=(lamT / lam)):
            k = k + 1
            t_old = S[-1]
            S.append(t)
            intervals.append(t - t_old)
    interval_a = np.array(intervals)
    interval_a = interval_a
    print("min: ", min(interval_a))
    print("max: ", max(interval_a))
    print("avg: ", np.mean(interval_a))
    current_time = time.time()
    times = []
    times.append(current_time)
    delays = [interval_a[0]]
    sum_delays = interval_a[0]
    for i in range(len(interval_a)):
        current_time += interval_a[i]
        times.append(current_time)
        sum_delays += interval_a[i]
        delays.append(sum_delays)
    with open(name, 'w') as f:
        for item in times:
            f.write("%s\n" % item)
    f.close()
    return times, delays


if load_pattern == "constant":
    load_value = float(args[3])
    intervals = constant(load_value)
else:
    load_pattern_file_name = args[3]
    intervals,l = from_load_pattern(load_pattern_file_name, 5, 1)

########################## Sending the requests using the asyncio library. ##########################
# Here is a very useful page to understand the concurrency better https://realpython.com/python-concurrency/
# In out application, we can send requests as a specific users too.
# Therefore, here we first send a request to the login page and login as a specific user and then access to the page we desire.
async def do_find_one(wait_time, counter, url, username = "premium", passwd = "1234", load_file_name = "load_compute.txt", response_file_name = "response_compute.txt", LD_ID=2):

    custom_header = {"Accept": "*/*", "Connection": "keep-alive", "Cache-Control": "max-age=0",
                     "Upgrade-Insecure-Requests": "1", "Origin": url,
                     "Content-Type": "application/x-www-form-urlencoded",
                     "Referer": url}
    d = dict()
    d["username"] = username
    d["passwd"] = passwd
    jar = aiohttp.CookieJar(unsafe=True)
    await asyncio.sleep(wait_time)
    start = time.time()
    async with aiohttp.ClientSession(trust_env=True,headers=custom_header,requote_redirect_url=True, cookie_jar=jar) as session:
        sent_text = dict()
        async with session.post(login_url, data=d) as response:
            sent_text["timesentout"] = start
            sent_text["timestamp"] = time.time()
            sent_text["ID"] = counter
            sent_text["LD_ID"] = LD_ID
            with open(load_file_name,"a") as load_file:
                load_file.write(str(sent_text)+"\n")
                load_file.flush()
            load_file.close()
            document = await response.text()
            print(document)
    end = time.time()
    sent_response = dict()
    sent_response["timestamp"] = str(time.time())
    sent_response["response"] = (end - start)
    sent_response["requestID"] = counter
    sent_response["LD_ID"] = LD_ID
    if (word1 in document):
        sent_response["Node"] = 1
    elif (word2 in document):
        sent_response["Node"] = 2
    else:
        sent_response["Node"] = 0
    if ("429 Too Many Requests" in document):
        sent_response["code"] = 429
    else:
        sent_response["code"] = 200
    print("start:{} response time:{}".format(str(start), str(end - start)))
    if (sent_response["response"] > 0):
        with open(response_file_name, "a") as response_file:
            response_file.write(str(sent_response) + "\n")
            response_file.flush()
    response_file.close()
    await session.close()


############################# The main function in which we start the load generator.###################
# In the main function all the coroutines that are responsible to send a request in speccific timestep
# (after specific delay) are generated. The main function ends whenever all coroutines are done their task.
async def main():
    print("This is the IP and port values:",IP_port)
    tasks = []
    print("max waiting time:{}".format(max(intervals)))
    counter = 0
    for j in intervals:
        counter += 1
        current_time = time.time()
        task = asyncio.ensure_future(do_find_one(j, counter,url,username,passwd,load_file_name,response_file_name,LD_ID))
        tasks.append(task)

    print("gather start")
    await asyncio.gather(*tasks, return_exceptions=True)
    print("gather end")

############For python lower or equal to 3.10
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()

############For python higher than 3.10
# if __name__ == '__main__':
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         pass
