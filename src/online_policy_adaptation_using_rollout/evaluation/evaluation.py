from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
import gym
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_1_env.routing_env import RoutingEnv
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_2_env.routing_env import RoutingEnv
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_3_env.routing_env import RoutingEnv

class EvaluateBasePolicy():
    def __init__(self, path_to_base_model, model_name, target_path_to_save_evalaution_results, episode_lenth, env_name,
                 path_to_system_model):
        self.path_to_base_model = path_to_base_model
        self.model_name = model_name
        self.target_path_to_save_evalaution_results = target_path_to_save_evalaution_results
        self.episode_lenth = episode_lenth
        self.env_name = env_name
        self.path_to_system_model = path_to_system_model

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

    # def evaluate_with_rollout(self):
