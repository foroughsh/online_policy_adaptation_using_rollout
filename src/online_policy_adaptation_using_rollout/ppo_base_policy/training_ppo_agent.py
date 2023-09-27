import random
import torch
import numpy as np
from stable_baselines3.common.monitor import Monitor
import gym
from stable_baselines3.common.callbacks import BaseCallback
import time
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

########## To train the agent for different scenarios, just import the environment for that scenario.
# Please note that for when we load the gym env in the main function we need to revise the version numbre based on the scenario number.
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_1_env.routing_env import RoutingEnv
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_2_env.routing_env import RoutingEnv
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_3_env.routing_env import RoutingEnv
class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, seed: int, model, env, path_to_models, path_to_results, policy_name="self_routing_", verbose=0, eval_every: int = 200):
        super(CustomCallback, self).__init__(verbose)
        self.iter = 0
        self.eval_every = eval_every
        self.seed = seed
        self.env = env
        self.path_to_models = path_to_models
        self.path_to_results = path_to_results
        self.model = model
        self.policy_name = policy_name

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print(f"Training iteration: {self.iter}")
        if (self.iter+1) % self.eval_every == 0:
            start = time.time()
            state = self.env.reset()
            s = state
            N = 1
            max_horizon = 7
            overall_rewards = []
            state_actions = []
            for i in range(N):
                done = False
                t = 0
                rewards = []
                actions = []
                infos = []
                while not done and t <= max_horizon:
                    print("state: ", s)
                    action_masks = get_action_masks(self.env)
                    print("action_mask:", action_masks)
                    a, _ = self.model.predict(s, deterministic=True, action_masks=action_masks)
                    print("current action for ",s," is ", a)
                    state_actions.append((s, a))
                    s, r, done, info = self.env.step(a)
                    print("next state: ", s)
                    print("reward: ", r)
                    print("info:", info)
                    print(50*"-")
                    rewards.append(round(r,3))
                    actions.append(a)
                    t+= 1
                    infos.append(info)
                overall_rewards.append(np.sum(rewards))
            avg_R = np.mean(overall_rewards)
            with open(self.path_to_results + f"avg_reward_{self.seed}.txt", 'a') as file:
                line = str(avg_R)+ "," + str(rewards) + "," + str(state) + "," + str(s) + "," + str(actions) + "\n"
                file.write(line)
            file.close()
            with open(self.path_to_results + f"infos{self.seed}.txt", 'a') as file:
                line = str(infos) + "\n"
                file.write(line)
            file.close()
            print(f"[EVAL] Training iteration: {self.iter}, Average R:{avg_R},\n list of (load, action): {state_actions}")
            self.env.reset()
            end = time.time()
            print(f"[EVAL] Training time: {end-start}")

        if self.iter % (self.eval_every*1) == 0:
            self.model.save(self.path_to_models + self.policy_name + str(self.seed)+"_"+ str(self.iter))
        self.iter += 1

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

class PPOBase():

    def __init__(self,environment_name, path_to_system_model, num_neurons_per_hidden_layer, num_layers, seed, path_to_save_models,
                 path_to_save_results, verbose, steps_between_updates, batch_size, learning_rate, gamma, ent_coef,
                 clip_range, num_training_timesteps, device, policy_name="self_routing"):
        self.environment_name = environment_name
        self.path_to_system_model = path_to_system_model
        self.num_neurons_per_hidden_layer = num_neurons_per_hidden_layer
        self.num_layers = num_layers
        self.seed = seed
        self.path_to_save_models = path_to_save_models
        self.path_to_save_results = path_to_save_results
        self.verbose = verbose
        self.steps_between_updates = steps_between_updates
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.num_training_timesteps = num_training_timesteps
        self.device = device
        self.policy_name = policy_name

    def train_ppo_base(self):
        # For other senarios the version of the environment is different.
        eval_env = gym.make(self.environment_name, path_to_system_model=self.path_to_system_model)
        env = gym.make(self.environment_name, path_to_system_model=self.path_to_system_model)
        env = Monitor(env)
        env.reset()

        action_masks = get_action_masks(env)
        # Hparams
        policy_kwargs = dict(net_arch=[self.num_neurons_per_hidden_layer] * self.num_layers)

        # Set seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Train
        model = MaskablePPO("MlpPolicy", env, verbose=self.verbose,
                    policy_kwargs=policy_kwargs, n_steps=self.steps_between_updates,
                    batch_size=self.batch_size, learning_rate=self.learning_rate, seed=self.seed,
                    device=self.device, gamma=self.gamma, ent_coef=self.ent_coef, clip_range=self.clip_range)

        cb = CustomCallback(seed=self.seed, env=eval_env, model=model, path_to_models=self.path_to_save_models,
                            path_to_results=self.path_to_save_results, eval_every=2, policy_name=self.policy_name)

        start = time.time()
        model.learn(total_timesteps=self.num_training_timesteps, callback=cb)
        end = time.time()

        return end-start
