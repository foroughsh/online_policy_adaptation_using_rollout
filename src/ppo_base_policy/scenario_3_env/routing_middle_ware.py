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

    def __init__(self):
        self.path = "/Users/foro/PycharmProjects/NOMS2024_rollouts/s6/system_model/"
        try:
            self.delay_model = joblib.load(self.path + "system_model.joblib")
        except:
            self.delay_model = joblib.load("/Users/kimham/Dropbox/NOMS24/SC1/delays_RF_model.joblib")
        self.state_size = 3
        self.state = np.zeros(self.state_size, dtype=float)
        ######actions#######################
        self.cpu1 = 1
        self.p1 = 1
        self.l1 = 5
        self.cl1 = self.l1
        self.b1 = 0
        #######initial response times##############
        [self.d1] = self.delay_model.predict([[self.cpu1, self.p1, self.l1, self.cl1]])
        #######state######################
        self.state[0] = self.l1
        self.state[1] = self.p1
        self.state[2] = self.cpu1

        self.state_counter = 0
        self.LD_counter = 0

    def get_state(self):
        return self.state

    def read_state_from_system(self, action):
        configuration = np.array([self.p1, self.cpu1])

        new_configuration = action_to_next_config(action, configuration)
        self.p1 = new_configuration[0]
        self.cpu1 = new_configuration[1]
        random_number = random.random()
        if (self.LD_counter)%20==0:
            if random_number<0.33:
                self.l1 = 5
            elif random_number<0.66:
                self.l1 = 10
            else:
                self.l1 = 15
        self.cl1 = self.l1
        [self.d1] = self.delay_model.predict([[self.cpu1, self.p1, self.l1, self.cl1]])

        self.state[0] = self.l1
        self.state[1] = self.p1
        self.state[2] = self.cpu1

        self.state_counter += 1
        self.LD_counter += 1
        return self.state, self.d1

    def reset(self):

        self.cl1 = self.l1

        self.p1 = random.randint(0, 1) / 5
        self.cpu1 = random.randint(1, 5)
        self.cl1 = self.l1

        [self.d1] = self.delay_model.predict(
            [[self.cpu1, self.p1, self.l1, self.cl1]])

        self.state[0] = self.l1
        self.state[1] = self.p1
        self.state[2] = self.cpu1

        return self.state

    def reset_to_specific_state(self, l1, p1, c1):
        self.LD_counter = 1
        self.cpu1 = c1
        self.p1 = p1
        self.l1 = l1
        self.cl1 = self.l1

        #######initial response times##############
        [self.d1] = self.delay_model.predict(
            [[self.cpu1, self.p1, self.l1, self.cl1]])
        #######state######################
        self.state[0] = self.l1
        self.state[1] = self.p1
        self.state[2] = self.cpu1

        return self.state

    def set_state(self, l1, p1, cpu1):
        self.state = [l1, p1, cpu1]
        self.p1 = p1
        self.cpu1 = cpu1
        self.l1 = l1
        return self.state

# test_routing = RoutingMiddleWare()
# for i in range(0,20):
#     print("State before the action: ", test_routing.state)
#     action = [random.randint(0,2),random.randint(0,2)]
#     state, d1= test_routing.read_state_from_system(action)
#     print(action,state, d1)
