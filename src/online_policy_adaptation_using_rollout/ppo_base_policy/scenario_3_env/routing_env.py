from typing import Tuple
import gym
import numpy as np
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_3_env.routing_middle_ware import RoutingMiddleWare
from typing import List

path = "/"

THERESHOL_D1 = 0.115
def reward_function(d1, cpu1,p):
    d1 = round(d1,3)
    reward = 0
    if (d1<=THERESHOL_D1):
        reward = (1-cpu1/5) * (1-p)
    else:
        if (d1 > THERESHOL_D1):
            reward = THERESHOL_D1 - d1
    return round(reward,3)

class RoutingEnv(gym.Env):

    def __init__(self, path_to_system_model):
        self.observation_space = gym.spaces.Box(low=np.array([5, 0, 0]), dtype=np.float32,
                                                high=np.array([15, 1, 5]),
                                                shape=(3,))
        self.action_space = gym.spaces.MultiDiscrete([3,3])
        self.path_to_system_model = path_to_system_model
        self.middleware = RoutingMiddleWare(self.path_to_system_model)

        self.l1 = 5
        self.cl1 = self.l1

        self.d1 = 0

        self.cpu1 = 1
        self.p1 = 1

        self.s = [self.l1, self.p1, self.cpu1]
        self.t = 1
        self.reset()
        self.path_to_mask = "/Users/foro/PycharmProjects/NOMS2024_rollouts/sc5/data/Not_in_OP_before.txt"

    def action_masks(self) -> List[bool]:
        if self.p1==0:
            mask1 = [1, 0, 1]
        elif self.p1==1:
            mask1 = [1, 1, 0]
        else:
            mask1 = [1, 1, 1]

        if self.cpu1==1:
            mask2 = [1, 0, 1]
        elif self.cpu1==5:
            mask2 = [1, 1, 0]
        else:
            mask2 = [1, 1, 1]


        mask = [mask1, mask2]
        return mask

    def step(self, a: gym.spaces.MultiDiscrete) -> Tuple[np.ndarray, int, bool, dict]:
        info = {}
        done = False
        next_state, d1 = self.middleware.read_state_from_system(a)
        self.l1 = next_state[0]
        self.p1 = next_state[1]
        self.cpu1 = next_state[2]

        self.cl1 = self.l1

        reward = reward_function(d1, self.cpu1,self.p1)

        info["l1"] = self.l1
        info["cl1"] = self.cl1
        info["d1"] = d1
        info["cpu1"] = self.cpu1
        info["p1"] = self.p1

        self.d1 = d1

        self.s = next_state

        info["r"] = reward
        # if self.t > 10:
        #     done = True
        self.t += 1

        return np.array([self.l1, self.p1, self.cpu1]), reward, done, info

    def reset(self) -> np.ndarray:
        self.t = 1

        self.middleware.reset()

        state, d1 = self.middleware.read_state_from_system([0,0])

        self.l1 = state[0]
        self.cl1 = self.l1
        self.p1 = state[1]
        self.cpu1 = state[2]

        self.d1 = d1

        self.s = state
        return np.array([self.l1, self.p1, self.cpu1])

    def reset_to_specific_state(self, l1, p1, c1) -> np.ndarray:
        self.t = 1

        self.middleware.reset_to_specific_state(l1, p1, c1)

        state, d1 = self.middleware.read_state_from_system([0,0])

        self.l1 = state[0]
        self.cl1 = self.l1
        self.p1 = state[1]
        self.cpu1 = state[2]

        self.d1 = d1

        self.s = state
        return np.array([self.l1, self.p1, self.cpu1])

    def set_state(self, l1, p1, cpu1):
        self.p1 = p1
        self.s = [l1, p1, cpu1]
        self.l1 = l1
        self.cpu1 = cpu1
        self.middleware.set_state(l1, p1, cpu1)
        return self.s

    def get_state(self):
        return self.s

# env = RoutingEnv()
# env.set_state(25, 0.4, 1)
# # for i in range(0,10):
# #     print("Step ----> ", i)
# #     action = [0,2]
# #     print("In the loop:",action)
# #     s, reward, done, info = env.step(action)
# #     print(s, reward, done, info)
# print(env.get_state())
# mask = env.action_masks()
# print(mask)
