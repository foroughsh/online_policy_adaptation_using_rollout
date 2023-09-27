from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
import gym
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_1_env.routing_env import RoutingEnv
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_2_env.routing_env import RoutingEnv
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_3_env.routing_env import RoutingEnv
import time
import numpy as np
import torch

class EvaluateBasePolicy():
    def __init__(self, path_to_base_model, model_name, target_path_to_save_evalaution_results, episode_lenth, env_name,
                 path_to_system_model):
        self.path_to_base_model = path_to_base_model
        self.model_name = model_name
        self.target_path_to_save_evalaution_results = target_path_to_save_evalaution_results
        self.episode_lenth = episode_lenth
        self.env_name = env_name
        self.path_to_system_model = path_to_system_model
        self.Timings = []

    def evaluate_with_base_or_offline(self):
        env = gym.make(self.env_name, path_to_system_model=self.path_to_system_model)
        env = Monitor(env)

        model = MaskablePPO.load(self.path_to_base_model + self.model_name)

        obs = env.reset()
        obs = env.set_state(1, 5)
        evaluation_step = 0

        rewards = []
        infos = []

        while evaluation_step <= self.episode_lenth:
            action_masks = get_action_masks(env)
            action, states_ = model.predict(obs, deterministic=True, action_masks=action_masks)
            print(action, states_)
            obs, reward, dones, info = env.step(action)
            print(obs, reward, dones, info)
            evaluation_step += 1
            rewards.append(reward)
            infos.append(info)
        return rewards, infos

    def n_step_rollout(self, current_state, steps_number, N, env, base_model, reward_so_far):
        print("The reward so far for state", current_state, " is: ", reward_so_far)
        start = time.time()
        if (steps_number == N):
            values = base_model.policy.predict_values(torch.tensor([current_state]))
            value = values.cpu().detach().numpy().squeeze()
            print("We return the reward from the leaf with the value of ", value + reward_so_far)
            print(50 * "=")
            return value + reward_so_far, -1
        else:
            action_masks = get_action_masks(env)
            counter = env.get_episode_counter()
            p1 = current_state[0]
            b1 = current_state[1]
            print("For the current state of ", current_state, " and action mask is: ", action_masks)
            actions = []
            for i in range(2):
                action = []
                for j in range(3):
                    if (action_masks[i][j] == 1):
                        action.append(j)
                actions.append(action)
            aa = []
            for i in range(len(actions[0])):
                for j in range(len(actions[1])):
                    aa.append([actions[0][i], actions[1][j]])
            print(aa)
            rewards = np.zeros((len(aa)))
            for i in range(len(aa)):
                print(10 * ".", "> The action is: ", aa[i])
                obs, reward, dones, info = env.step(aa[i])
                next_reward_so_far = reward_so_far + reward
                print("The reward so far is: ", next_reward_so_far)
                print(10 * ">", "calling rollout again")
                rollout_reward, _ = self.n_step_rollout(obs, steps_number + 1, N, env, base_model, next_reward_so_far)
                rewards[i] = rollout_reward
                s = env.set_state(p1, b1)
                print("The state of environment is now back to ", s)
                env.set_episode_counter(counter)
            selected_action_by_rollout = np.argmax(rewards)
            end = time.time()
            print("**** The rollout time is :", end - start)
            self.Timings.append(end - start)
            return np.max(rewards), aa[selected_action_by_rollout]

    def evaluate_with_rollout(self):
        env = gym.make(self.env_name, path_to_system_model=self.path_to_system_model)
        env = Monitor(env)

        model = MaskablePPO.load(self.path_to_base_model + self.model_name)

        obs = env.reset()
        obs = env.set_state(1, 5)

        rewards = []

        reward_so_far = 0
        current_state = obs
        steps_number = 0
        N = 1
        episode_length = 8
        t = 0

        agg_reward = 0

        actions_from_rollout = []
        infos = []
        steps = []
        while (t < episode_length):
            start = time.time()
            max_reward, selected_action = self.n_step_rollout(current_state, steps_number, N, env, model, reward_so_far)
            next_obs, obtained_reward, dones, info = env.step(selected_action)
            end = time.time()
            steps.append(end - start)
            agg_reward += obtained_reward
            current_state = next_obs
            t += 1
            env.set_episode_counter(t)
            actions_from_rollout.append(selected_action)
            infos.append(info)
            rewards.append(obtained_reward)
        return rewards, infos