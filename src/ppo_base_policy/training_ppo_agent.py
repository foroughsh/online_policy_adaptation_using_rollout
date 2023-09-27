import random
import torch
import numpy as np
from stable_baselines3.common.monitor import Monitor
import gym
from two_services_env.env.routing_env import RoutingEnv
from stable_baselines3.common.callbacks import BaseCallback
import time
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import sys
import argparse

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, seed: int, env, path_to_models, path_to_results, verbose=0, eval_every: int = 200):
        super(CustomCallback, self).__init__(verbose)
        self.iter = 0
        self.eval_every = eval_every
        self.seed = seed
        self.env = env
        self.path_to_models = path_to_models
        self.path_to_results = path_to_results

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
            state = self.env.reset_to_specific_state(0.4,5)
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
                    a, _ = model.predict(s, deterministic=True, action_masks=action_masks)
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
            model.save(self.path_to_models + "self_routing_" + str(self.seed)+"_"+ str(self.iter))
        self.iter += 1

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

if __name__ == '__main__':
    eval_env = gym.make("routing-env-v2")
    env = gym.make("routing-env-v2")
    env = Monitor(env)
    env.reset()

    action_masks = get_action_masks(env)
    print("Action mask: ", action_masks)

    args = sys.argv


    if (len(args) > 1):
        parser = argparse.ArgumentParser(description='Please check the code for options!')
        parser.add_argument("--seed", type=int, default=83)
        parser.add_argument("--target_path_for_models", type=str, default="../../../../trained_models/scenario_2/")
        parser.add_argument("--target_path_to_save_results", type=str, default="../../../../artifacts/scenario_2/")
        parser.add_argument("--num_neurons_per_hidden_layer", type=int, default=128)
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--steps_between_updates", type=int, default=512)
        parser.add_argument("--learning_rate", type=float, default=0.0005)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--num_training_timesteps", type=int, default=1000000)
        parser.add_argument("--step", default=100)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--num_training_timesteps", type=int, default=1000000)
        parser.add_argument("--verbose", type=int, default=0)
        parser.add_argument("--ent_coef", type=float, default=0.05)
        parser.add_argument("--clip_range", type=float, default=0.2)

        args = parser.parse_args()

        path_to_save_results = args.target_path_for_models
        path_to_save_models = args.target_path_to_save_results

        seed = args.seed
        num_neurons_per_hidden_layer = args.num_neurons_per_hidden_layer
        num_layers = args.num_layers
        steps_between_updates = args.steps_between_updates
        learning_rate = args.learning_rate
        batch_size = args.batch_size
        device = args.device
        gamma = args.gamma
        num_training_timesteps = args.num_training_timesteps
        verbose = args.verbose
        ent_coef = args.ent_coef
        clip_range = args.clip_range
    else:
        print(
            "Please enter the seed and other arguments. The list of options are as the following:\n"
            "\t--seed 83\n"
            "\t--target_path_for_models ../../trained_models/scenario_2/\n"
            "\t--target_path_to_save_results ../../artifacts/scenario_2/\n"
            "\t--num_neurons_per_hidden_layer 128\n"
            "\t--num_layers 3\n"
            "\t--steps_between_updates 512\n"
            "\t--learning_rate 0.0005\n"
            "\t--batch_size 64\n"
            "\t--device cpu\n"
            "\t--gamma 0.99\n"
            "\t--num_training_timesteps 1000000"
            "\t--verbose 0\n"
            "\t--ent_coef 0.05\n"
            "\t--clip_range 0.2"
        )

    # Hparams
    policy_kwargs = dict(net_arch=[num_neurons_per_hidden_layer] * num_layers)

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cb = CustomCallback(seed=seed, env=eval_env, path_to_models=path_to_save_models,
                        path_to_results=path_to_save_results, eval_every=2)

    # Train
    model = MaskablePPO("MlpPolicy", env, verbose=verbose,
                policy_kwargs=policy_kwargs, n_steps=steps_between_updates,
                batch_size=batch_size, learning_rate=learning_rate, seed=seed,
                device="cpu", gamma=gamma, ent_coef=ent_coef, clip_range=clip_range)
    start = time.time()
    model.learn(total_timesteps=num_training_timesteps, callback=cb)
    end = time.time()

    print("Training time: ", end - start)
