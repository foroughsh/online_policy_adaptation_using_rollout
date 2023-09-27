import random
import numpy as np
import joblib

np.set_printoptions(suppress=True)

def action_to_next_config(actions, configurations):
    new_configuration = np.zeros((configurations.shape))
    for i in range(0, 4):
        if i < len(configurations)/2:
            if actions[i] == 0:
                new_configuration[i] = configurations[i]
            elif actions[i] == 1:
                new_configuration[i] = configurations[i] - 0.2
            elif actions[i] == 2:
                new_configuration[i] = configurations[i] + 0.2
            else:
                raise ValueError("bug")
            if new_configuration[i]<0:
                new_configuration[i]= 0
            elif new_configuration[i]>1:
                new_configuration[i] = 1
        else:
            if actions[i] == 0:
                new_configuration[i] = configurations[i]
            elif actions[i] == 1:
                new_configuration[i] = configurations[i] - 1000
            elif actions[i] == 2:
                new_configuration[i] = configurations[i] + 1000
            else:
                raise ValueError("bug")
            if new_configuration[i]<2000:
                new_configuration[i]= 2000
            elif new_configuration[i]>5000:
                new_configuration[i] = 5000
    return new_configuration


class RoutingMiddleWare():

    def __init__(self, system_model_path):
        self.path = system_model_path
        self.delay_model = joblib.load(self.path + "delays_RF_model.joblib")
        self.state_size = 4
        self.state = np.zeros(self.state_size, dtype=float)
        ######actions#######################
        self.cpu1 = 2
        self.cpu2 = 2
        self.p1 = 1
        self.p2 = 1
        self.l1 = 4
        self.l2 = 15
        self.cl1 = self.l1
        self.cl2 = self.l2
        #######initial response times##############
        [[self.d1, self.d2]] = self.delay_model.predict([[self.cpu1, self.cpu2, self.p1, self.p2, self.l1, self.cl1, self.l2, self.cl2]])
        #######state######################
        self.state[0] = self.p1
        self.state[1] = self.p2
        self.state[2] = self.cpu1
        self.state[3] = self.cpu2

        self.state_counter = 0
        self.LD_counter = 0

    def get_state(self):
        return self.state

    def read_state_from_system(self, action):
        configuration = np.array([self.p1, self.p2, self.cpu1, self.cpu2])

        new_configuration = action_to_next_config(action, configuration)
        self.p1 = new_configuration[0]
        self.p2 = new_configuration[1]
        self.cpu1 = new_configuration[2]
        self.cpu2 = new_configuration[3]

        self.cl1 = self.l1
        self.cl2 = self.l2

        [[self.d1, self.d2]] = self.delay_model.predict([[self.cpu1, self.cpu2, self.p1, self.p2, self.l1, self.cl1, self.l2, self.cl2]])

        self.state[0] = self.p1
        self.state[1] = self.p2
        self.state[2] = self.cpu1
        self.state[3] = self.cpu2

        self.state_counter += 1
        self.LD_counter += 1
        return self.state, self.d1, self.d2

    def reset(self):

        self.state_counter = 0
        self.l1 = 4
        self.l2 = 15
        self.cl1 = self.l1
        self.cl2 = self.l2

        self.p1 = random.randint(0, 1) / 5
        self.p2 = random.randint(0, 1) / 5
        self.cpu1 = random.randint(2, 5) * 1000
        self.cpu2 = random.randint(2, 5) * 1000
        self.cl1 = self.l1
        self.cl2 = self.l2

        [[self.d1, self.d2]] = self.delay_model.predict(
            [[self.cpu1, self.cpu2, self.p1, self.p2, self.l1, self.cl1, self.l2, self.cl2]])

        self.state[0] = self.p1
        self.state[1] = self.p2
        self.state[2] = self.cpu1
        self.state[3] = self.cpu2

        return self.state

    def reset_to_specific_state(self, p1, p2, c1, c2):
        self.state_counter = 0
        self.cpu1 = c1
        self.cpu2 = c2
        self.p1 = p1
        self.p2 = p2
        self.l1 = 4
        self.cl1 = self.l1
        self.l2 = 15
        self.cl2 = self.l2

        #######initial response times##############
        [[self.d1, self.d2]] = self.delay_model.predict(
            [[self.cpu1, self.cpu2, self.p1, self.p2, self.l1, self.cl1, self.l2, self.cl2]])
        #######state######################
        self.state[0] = self.p1
        self.state[1] = self.p2
        self.state[2] = self.cpu1
        self.state[3] = self.cpu2

        return self.state

    def set_state(self, p1, p2, cpu1, cpu2):
        print("(SYS) Received state is: ", [p1, p2, cpu1, cpu2])
        self.state = [p1, p2, cpu1, cpu2]
        self.p1 = p1
        self.p2 = p2
        self.cpu1 = cpu1
        self.cpu2 = cpu2
        print("(SYS) Current state: ", self.state)
        return self.state

# test_routing = RoutingMiddleWare()
# # new_cpu, new_routing, new_blocking = 1000, 0.2, 0.2
# # for i in range(8):
# #     new_cpu, new_routing, new_blocking = action_to_next_config(i, new_cpu, new_routing, new_blocking)
# #     print(new_cpu, new_routing, new_blocking)
# for i in range(0,10):
#     print("State before the action: ", test_routing.state)
#     action = [0,1,2,1]
#     state, d1, d2= test_routing.read_state_from_system(action)
#     print(action,state, d1, d2)
