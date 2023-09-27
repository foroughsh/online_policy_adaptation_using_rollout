import random
import numpy as np
import joblib

np.set_printoptions(suppress=True)

def action_to_next_config(actions, configurations):
    new_configuration = np.zeros((configurations.shape))
    for i in range(0, 2):
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
                new_configuration[i] = configurations[i] - 1
            elif actions[i] == 2:
                new_configuration[i] = configurations[i] + 1
            else:
                raise ValueError("bug")
            if new_configuration[i]<1:
                new_configuration[i]= 1
            elif new_configuration[i]>5:
                new_configuration[i] = 5
    return new_configuration


class RoutingMiddleWare():

    def __init__(self, path_to_system_model):
        self.path = path_to_system_model
        self.delay_model = joblib.load(self.path)
        self.state_size = 2
        self.state = np.zeros(self.state_size, dtype=float)
        ######actions#######################
        self.cpu1 = 1
        self.p1 = 1
        self.l1 = 5
        self.cl1 = self.l1
        self.b1 = 0
        #######initial response times##############
        #### feature order in our system model: ["c1", "p11", "l1", "cl1"]
        [self.d1] = self.delay_model.predict([[self.cpu1, self.p1, self.l1, self.cl1]])
        #######state######################
        self.state[0] = self.p1
        self.state[1] = self.cpu1

        self.state_counter = 0
        self.LD_counter = 0

    def get_state(self):
        return self.state

    def read_state_from_system(self, action):
        configuration = np.array([self.p1, self.cpu1])

        new_configuration = action_to_next_config(action, configuration)
        self.p1 = new_configuration[0]
        self.cpu1 = new_configuration[1]

        self.cl1 = self.l1

        [self.d1] = self.delay_model.predict([[self.cpu1, self.p1, self.l1, self.cl1]])

        self.state[0] = self.p1
        self.state[1] = self.cpu1

        self.state_counter += 1
        self.LD_counter += 1
        return self.state, self.d1

    def reset(self):

        self.state_counter = 0
        self.l1 = 5
        self.cl1 = self.l1

        self.p1 = random.randint(0, 1) / 5
        self.cpu1 = random.randint(1, 5)
        self.cl1 = self.l1

        [self.d1] = self.delay_model.predict(
            [[self.cpu1, self.p1, self.l1, self.cl1]])

        self.state[0] = self.p1
        self.state[1] = self.cpu1

        return self.state

    def reset_to_specific_state(self, p1, c1):
        self.state_counter = 0
        self.cpu1 = c1
        self.p1 = p1
        self.l1 = 5
        self.cl1 = self.l1

        #######initial response times##############
        [self.d1] = self.delay_model.predict(
            [[self.cpu1, self.p1, self.l1, self.cl1]])
        #######state######################
        self.state[0] = self.p1
        self.state[1] = self.cpu1

        return self.state

    def set_state(self, p1, cpu1):
        self.state = [p1, cpu1]
        self.p1 = p1
        self.cpu1 = cpu1
        return self.state

# test_routing = RoutingMiddleWare()
# for i in range(0,10):
#     print("State before the action: ", test_routing.state)
#     action = [0,1,2,1]
#     state, d1, d2= test_routing.read_state_from_system(action)
#     print(action,state, d1, d2)
