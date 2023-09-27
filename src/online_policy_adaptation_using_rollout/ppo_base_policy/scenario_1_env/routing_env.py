from typing import Tuple
import gym
import numpy as np
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_1_env.routing_middle_ware import RoutingMiddleWare
from typing import List

path = "/"

THERESHOL_D1 = 0.95
THERESHOL_D2 = 0.115

def reward_function(d1, d2, cpu1, cpu2, p1, p2):
    if (d1<=THERESHOL_D1) and (d2<=THERESHOL_D2):
        overal_cpu = (cpu1 + cpu2)/10000
        reward = (1-overal_cpu) * (1-p1) * (1-p2)
    else:
        reward = 0
    return reward


class RoutingEnv(gym.Env):

    def __init__(self, path_to_system_model):
        self.path_to_system_model = path_to_system_model
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), dtype=np.float32,
                                                high=np.array([1, 1, 5000, 5000]),
                                                shape=(4,))
        self.action_space = gym.spaces.MultiDiscrete([3,3,3,3])

        self.middleware = RoutingMiddleWare(self.path_to_system_model)

        self.l1 = 4
        self.cl1 = 4
        self.l2 = 15
        self.cl2 = 15

        self.d1 = 0
        self.d2 = 0

        self.cpu1 = 2
        self.p1 = 1
        self.cpu2 = 2
        self.p2 = 1

        self.s = [self.p1, self.p2, self.cpu1, self.cpu2]
        self.t = 1
        self.reset()

    def action_masks(self) -> List[bool]:
        if self.p1==0:
            mask1 = [1, 0, 1]
        elif self.p1==1:
            mask1 = [1, 1, 0]
        else:
            mask1 = [1, 1, 1]

        if self.p2==0:
            mask2 = [1, 0, 1]
        elif self.p2==1:
            mask2 = [1, 1, 0]
        else:
            mask2 = [1, 1, 1]

        if self.cpu1==2000:
            mask3 = [1, 0, 1]
        elif self.cpu1==5000:
            mask3 = [1, 1, 0]
        else:
            mask3 = [1, 1, 1]

        if self.cpu2==2000:
            mask4 = [1, 0, 1]
        elif self.cpu2==5000:
            mask4 = [1, 1, 0]
        else:
            mask4 = [1, 1, 1]
        mask = [mask1, mask2, mask3, mask4]
        return mask


    def step(self, a: gym.spaces.MultiDiscrete) -> Tuple[np.ndarray, int, bool, dict]:
        info = {}
        done = False
        next_state, d1, d2 = self.middleware.read_state_from_system(a)
        p1 = next_state[0]
        p2 = next_state[1]
        cpu1 = next_state[2]
        cpu2 = next_state[3]

        cl1 = self.l1
        cl2 = self.l2

        reward = reward_function(d1, d2, cpu1, cpu2, p1, p2)

        info["l1"] = 4
        info["cl1"] = cl1
        info["l2"] = 15
        info["cl2"] = cl2
        info["d1"] = d1
        info["d2"] = d2
        info["cpu1"] = cpu1
        info["cpu2"] = cpu2
        info["p1"] = p1
        info["p2"] = p2

        self.l1 = 4
        self.cl1 = cl1
        self.l2 = 15
        self.cl2 = cl2

        self.d1 = d1
        self.d2 = d2

        self.cpu1 = cpu1
        self.cpu2 = cpu2
        self.p1 = p1
        self.p2 = p2
        self.s = next_state

        info["r"] = reward
        self.t += 1

        return np.array([self.p1, self.p2, self.cpu1, self.cpu2]), reward, done, info

    def reset(self) -> np.ndarray:
        self.t = 1

        self.middleware.reset()

        state, d1, d2 = self.middleware.read_state_from_system([0,0,0,0])

        self.l1 = 4
        self.cl1 = 4
        self.l2 = 15
        self.cl2 = 15
        self.p1 = state[0]
        self.p2 = state[1]
        self.cpu1 = state[2]
        self.cpu2 = state[3]

        self.d1 = d1
        self.d2 = d2

        self.s = state
        return np.array([self.p1, self.p2, self.cpu1, self.cpu2])

    def reset_to_specific_state(self, p1, p2, c1, cp2) -> np.ndarray:
        self.t = 1

        self.middleware.reset_to_specific_state(p1, p2, c1, cp2)

        state, d1, d2 = self.middleware.read_state_from_system([0,0,0,0])

        self.l1 = 4
        self.cl1 = 4
        self.l2 = 15
        self.cl2 = 15
        self.p1 = state[0]
        self.p2 = state[1]
        self.cpu1 = state[2]
        self.cpu2 = state[3]

        self.d1 = d1
        self.d2 = d2

        self.s = state
        return np.array([self.p1, self.p2, self.cpu1, self.cpu2])

    def set_state(self, p1, p2, cpu1, cpu2):
        print("(EVN) Received state is: ", [p1, p2, cpu1, cpu2])
        self.p1 = p1
        self.p2 = p2
        self.s = [p1, p2, cpu1, cpu2]
        print("(EVN) Current state: ", self.s)
        self.middleware.set_state(p1, p2, cpu1, cpu2)
        return self.s

    def get_episode_counter(self):
        return self.t

    def set_episode_counter(self, t):
        self.t = t

# env = RoutingEnv()
# for i in range(0,10):
#     print("Step ----> ", i)
#     action = [0,2,1,0]
#     print("In the loop:",action)
#     s, reward, done, info = env.step(action)
#     print(s, reward, done, info)
