"""
Script with functions for running different algorithms for scenario 1
"""
from typing import List, Union, Tuple, Any
import time
import math
import numpy as np
import numpy.typing as npt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from algorithms.value_iteration_scenario_1 import value_iteration, greedy_policy
from algorithms.policy_iteration_scenario_1 import policy_iteration, \
    policy_evaluation
from algorithms.rollouts_scenario_1 import exact_one_step_rollout_policy, \
    monte_carlo_policy_evaluation, \
    monte_carlo_l_step_rollout_policy, monte_carlo_l_step_multiagent_rollout_policy
from system_model.system_model_scenario_1 import reward_tensor, \
    transition_tensor, transition_function, state_space, \
    action_space, services, edges, nodes, precompute_delays, reward_function, convert_env_a_to_dp_a, \
    convert_dp_s_to_env_s
from sc1.rollout.two_services_env.env.routing_env import RoutingEnv
import random


def newton_step_data(base_pi: Union[List, MaskablePPO], states: List, actions: List, gamma: float,
                     S: list, N: list, E: list, p_max: float, c_max: float, initial_state: int,
                     batch_size: int, V_base: List, l: int,
                     deterministic_model: bool, delay_table: List, alt_pi, alt_V, env=None) \
        -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Implements the rollout Newton step in a data-driven fashion through monte-carlo evaluation of the value function

    :param base_pi: the base policy
    :param states: the state space
    :param actions: the action space
    :param gamma: the discount factor
    :param S: the set of services
    :param N: the set of nodes
    :param E: the set of edges
    :param p_max: the max value for the routing action
    :param c_max: the max value for the scaling action
    :param initial_state: the initial state
    :param batch_size: the batch size
    :param V_base: the base value function
    :param l: the lookahead horizon
    :param deterministic_model: boolean flag indicating whether the model is deterministic or not
    :param delay_table: a table with delay predictions
    :param alt_pi: an alternative base policy for evaluation
    :param alt_V: an alternative base value function for evaluation
    :param env: the RL environment
    :return: value of initial state with two differnet rollout policies and two base policies, slopes of the
             Bellman operators, intercepts of the Bellman operators
    """
    s = initial_state
    rollout_pi, V_rollout, rollout_a = monte_carlo_l_step_rollout_policy(base_pi=base_pi, s=s, gamma=gamma,
                                                                         batch_size=batch_size, states=states,
                                                                         actions=actions, c_max=c_max,
                                                                         p_max=p_max, N=N,
                                                                         E=E, S=S, V_base=V_base, l=l,
                                                                         deterministic_model=deterministic_model,
                                                                         delay_table=delay_table, env=env)
    V_base_2 = V_base.copy()
    V_base_2[s] = 0.5
    rollout_pi_2, V_rollout_2, rollout_a_2 = monte_carlo_l_step_rollout_policy(base_pi=base_pi, s=s, gamma=gamma,
                                                                               batch_size=batch_size, states=states,
                                                                               actions=actions, c_max=c_max,
                                                                               p_max=p_max, N=N,
                                                                               E=E, S=S, V_base=V_base_2, l=l,
                                                                               deterministic_model=deterministic_model,
                                                                               delay_table=delay_table, env=env)
    alt_pi_2 = alt_pi.copy()
    obs = convert_dp_s_to_env_s(s=states[s])
    env.set_state(p1=obs[0], p2=obs[1], cpu1=obs[2], cpu2=obs[3])
    action_masks = get_action_masks(env)
    base_a, _ = base_pi.predict(obs, deterministic=True, action_masks=action_masks)
    base_a = convert_env_a_to_dp_a(a=list(base_a), delta_p=delta_p, delta_c=delta_c)
    base_a = actions.index(tuple(base_a))
    V_base_3 = monte_carlo_policy_evaluation(policy=alt_pi_2, states=states, actions=actions, gamma=gamma,
                                             batch_size=batch_size, N=N, E=E, c_max=c_max, p_max=p_max, S=S,
                                             delay_table=delay_table, delta_c=1,
                                             delta_p=0.2, modified_s=s, modified_a=base_a, eval_state=None, env=env)

    T_rollout_slope = round((V_rollout[s] - V_rollout_2[s]) / (V_rollout[s] - 0.5), 3)
    T_rollout_intercept = round(V_rollout[s] - T_rollout_slope * V_rollout[s], 3)
    T_base_slope = round((V_base[s] - V_base_3[s]) / (V_base[s] - alt_V[s]), 3)
    T_base_intercept = round(V_base[s] - T_base_slope * V_base[s], 3)

    return V_rollout[s], V_rollout_2[s], V_base[s], V_base_3[
        s], T_rollout_slope, T_rollout_intercept, T_base_slope, T_base_intercept


def run_base_policy_online(base_pi: Union[List, MaskablePPO], states: List, actions: List,
                           S: list, N: list, E: list, p_max: float, c_max: float, initial_state: int,
                           horizon: int, V_base: List,
                           seeds: List[int],
                           delay_table: List, num_episodes: int = 1,
                           delta_p: float = None, delta_c: float = None, env=None) -> Tuple[List[float, List[float]]]:
    """
    Runs the base policy online

    :param base_pi: the base policy to run
    :param states: the state space
    :param actions: the action space
    :param S: the list of services
    :param N: the list of nodes
    :param E: the list of edges
    :param p_max: the maximum routing action value
    :param c_max: the maximum scaling action value
    :param initial_state: the initial state
    :param horizon: the horizon for online play
    :param V_base: the value function of the base policy
    :param seeds: the random seeds
    :param delay_table: the table with predicted delays
    :param num_episodes: the number of episodes to run
    :param delta_p: the discretization step of the routing action
    :param delta_c: the discretization step of the scaling action
    :param env: the evaluation environment
    :return: the mean returns and standard deviations
    """
    reward_trajectories = []
    for k in range(num_episodes):
        np.random.seed(seeds[k])
        random.seed(seeds[k])
        s = initial_state
        print(f"Starting monte-carlo online play, initial state: {s}, episode: {k}/{num_episodes}")
        rewards = []
        for i in range(horizon):
            if isinstance(base_pi, list) or isinstance(base_pi, np.ndarray):
                base_a = int(np.random.choice(np.arange(0, len(actions)), p=base_pi[s]))
                base_a = actions[base_a]
            else:
                obs = convert_dp_s_to_env_s(s=states[s])
                env.set_state(p1=obs[0], p2=obs[1], cpu1=obs[2], cpu2=obs[3])
                action_masks = get_action_masks(env)
                base_a, _ = base_pi.predict(obs, deterministic=True, action_masks=action_masks)
                base_a = convert_env_a_to_dp_a(a=list(base_a), delta_p=delta_p, delta_c=delta_c)
            base_a_idx = actions[actions.index(tuple(base_a))]
            r_base = reward_function(s=states[int(s)], a=base_a_idx, services=S, nodes=N,
                                     edges=E, c_max=c_max,
                                     p_max=p_max, states=states, delay_table=delay_table)
            rewards.append(r_base)
            print(f"s: {states[s]}, V_base[s]:{V_base[s]}, "
                  f"base_pi[s]: {base_a}, "
                  f"r_base: {r_base}")
            s = transition_function(s=states[int(s)], a=base_a_idx, nodes=N, edges=E, c_max=c_max,
                                    p_max=p_max, services=S)
            s = states.index(s)
        reward_trajectories.append(rewards)
    means = []
    stds = []
    for i in range(len(reward_trajectories[0])):
        rewards = []
        for k in range(num_episodes):
            rewards.append(reward_trajectories[k][i])
        rewards = np.array(rewards)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards) * (1 / math.sqrt(num_episodes)))
    return means, stds


def run_monte_carlo_one_step_online_play(base_pi: Union[List, MaskablePPO], states: List, actions: List, gamma: float,
                                         S: list, N: list, E: list, p_max: float, c_max: float, initial_state: int,
                                         horizon: int, batch_size: int, V_base: List, l: int,
                                         seeds: List[int], component_spaces,
                                         deterministic_model: bool, delay_table: List, num_episodes: int = 1,
                                         delta_p: float = None, delta_c: float = None, env=None,
                                         multiagent: bool = False) \
        -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Runs the rollout algorithm with monte carlo value evaluation

    :param base_pi: the base policy
    :param states: the state space
    :param actions: the action space
    :param gamma: the discount factor
    :param S: the set of services
    :param N: the set of nodes
    :param E: the set of edges
    :param p_max: the maximum routing action value
    :param c_max: the maximum scaling action value
    :param initial_state: the initial state
    :param horizon: the horizon for the evaluation
    :param batch_size: the batch size for the evaluation
    :param V_base: the value function of the base policy
    :param l: the lookahead optimization horizon
    :param seeds: the random seeds
    :param component_spaces: the action spaces of each action component
    :param deterministic_model: boolean flag indicating whether it is a deterministic model or not
    :param delay_table: a table with the predicted response times
    :param num_episodes: the number of episodes to run
    :param delta_p: the discretization step of the routing action
    :param delta_c: the discretization step of the scaling action
    :param env: the environment for evaluation
    :param multiagent: a boolean flag indicating whether the multiagent rollout algorithm should be run
    :return: mean returns, std returns, mean times, std times
    """
    reward_trajectories = []
    time_trajectories = []
    for k in range(num_episodes):
        np.random.seed(seeds[k])
        random.seed(seeds[k])
        s = initial_state
        print(f"Starting monte-carlo online play, initial state: {s}, episode: {k}/{num_episodes}, l: {l}")
        rewards = []
        rollout_times = []
        for i in range(horizon):
            start = time.time()
            if not multiagent:
                rollout_pi, V_rollout, rollout_a = monte_carlo_l_step_rollout_policy(base_pi=base_pi, s=s, gamma=gamma,
                                                                                     batch_size=batch_size,
                                                                                     states=states,
                                                                                     actions=actions, c_max=c_max,
                                                                                     p_max=p_max, N=N,
                                                                                     E=E, S=S, V_base=V_base, l=l,
                                                                                     deterministic_model=deterministic_model,
                                                                                     delay_table=delay_table, env=env)
            if multiagent:
                rollout_pi, V_rollout, rollout_a = monte_carlo_l_step_multiagent_rollout_policy(
                    base_pi=base_pi, s=s, gamma=gamma, batch_size=batch_size, states=states, actions=actions,
                    c_max=c_max, p_max=p_max, N=N, E=E, S=S, V_base=V_base, l=l,
                    deterministic_model=deterministic_model,
                    delay_table=delay_table, env=env, component_spaces=component_spaces)
            end = time.time()
            rollout_times.append(end - start)

            if isinstance(base_pi, list) or isinstance(base_pi, np.ndarray):
                base_a = int(np.random.choice(np.arange(0, len(actions)), p=base_pi[s]))
                base_a = actions[base_a]
            else:
                obs = convert_dp_s_to_env_s(s=states[s])
                env.set_state(p1=obs[0], p2=obs[1], cpu1=obs[2], cpu2=obs[3])
                action_masks = get_action_masks(env)
                base_a, _ = base_pi.predict(obs, deterministic=False, action_masks=action_masks)
                base_a = convert_env_a_to_dp_a(a=list(base_a), delta_p=delta_p, delta_c=delta_c)
            r_rollout = reward_function(s=states[int(s)], a=actions[rollout_a], services=S, nodes=N, edges=E,
                                        c_max=c_max,
                                        p_max=p_max, states=states, delay_table=delay_table)
            r_base = reward_function(s=states[int(s)], a=actions[actions.index(tuple(base_a))], services=S, nodes=N,
                                     edges=E, c_max=c_max,
                                     p_max=p_max, states=states, delay_table=delay_table)
            rewards.append(r_rollout)
            print(f"s: {states[s]}, V_base[s]:{V_base[s]}, V_rollout[s]: {V_rollout[s]}, "
                  f"base_pi[s]: {base_a}, "
                  f"rollout_pi[s]: {actions[rollout_a]}, r_base: {r_base}, r_rollout: {r_rollout}")
            s = transition_function(s=states[int(s)], a=actions[rollout_a], nodes=N, edges=E, c_max=c_max,
                                    p_max=p_max, services=S)
            s = states.index(s)
        reward_trajectories.append(rewards)
        time_trajectories.append(rollout_times)
    means = []
    stds = []
    for i in range(len(reward_trajectories[0])):
        rewards = []
        for k in range(num_episodes):
            rewards.append(reward_trajectories[k][i])
        rewards = np.array(rewards)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))

    time_means = []
    time_stds = []
    for i in range(len(time_trajectories[0])):
        times = []
        for k in range(num_episodes):
            times.append(time_trajectories[k][i])
        times = np.array(times)
        time_means.append(np.mean(times))
        time_stds.append(np.std(times))

    return means, stds, time_means, time_stds


def run_exact_one_step_online_play(base_pi: List, states: List, actions: List, gamma: float, theta: float,
                                   S: list, N: list, E: list, p_max: float, c_max: float, initial_state: int,
                                   horizon: int, delay_table: List) -> None:
    """
    Exact one-step online play

    :param base_pi: the base policy
    :param states: the state space
    :param actions: the action space
    :param gamma: the discount factor
    :param theta: the theta for value iteration stopping critertion
    :param S: the ser of services
    :param N: the set of nodes
    :param E: the set of edges
    :param p_max: the maximum routing action value
    :param c_max: the maximum scaling action value
    :param initial_state: the initial state
    :param horizon: the horizon to run the evaluation
    :param delay_table: a table with the predicted response times
    :return: None
    """
    print("Creating reward tensor")
    R = reward_tensor(state_space=states, action_space=actions, services=S, nodes=N, edges=E, c_max=c_max, p_max=p_max,
                      delay_table=delay_table)
    print("Creating transition tensor")
    T = transition_tensor(state_space=states, action_space=actions, services=S, nodes=N, edges=E, p_max=p_max,
                          c_max=c_max)
    print("Evaluating base policy")
    V_base = policy_evaluation(policy=base_pi, num_states=len(states), gamma=gamma, theta=theta, T=T, R=R)
    s = initial_state
    print(f"Starting exact online play, initial state: {s}")
    for i in range(horizon):
        rollout_pi, V_rollout = exact_one_step_rollout_policy(base_pi=base_pi, V_base=V_base, s=s, gamma=gamma,
                                                              num_actions=len(actions), num_states=len(states), T=T,
                                                              R=R, theta=theta)
        print(f"s: {states[s]}, V_base[s]:{V_base[s]}, V_rollout[s]: {V_rollout[s]}, "
              f"base_pi[s]: {actions[np.argmax(base_pi[s])]}, "
              f"rollout_pi[s]: {actions[np.argmax(rollout_pi[s])]}")
        s = transition_function(s=states[int(s)], a=actions[np.argmax(rollout_pi[s])], nodes=N, edges=E, c_max=c_max,
                                p_max=p_max, services=S)
        s = states.index(s)


def run_vi(states: List, actions: List, gamma: float, S: list, N: list, E: list, p_max: float, c_max: float,
           theta: float, delay_table: List) -> None:
    """
    Runs the value iteration algorithm

    :param states: the state space
    :param actions: the action space
    :param gamma: the discount factor
    :param S: the set of services
    :param N: the set of nodes
    :param E: the set of edges
    :param p_max: the maximum value of the routing action
    :param c_max: the maximum value of the scaling action
    :param theta: the theta value to measure convergence
    :param delay_table: the table with the predicted response times of different services in different states
    :return: None
    """
    print("Creating reward tensor")
    R = reward_tensor(state_space=states, action_space=actions, services=S, nodes=N, edges=E, c_max=c_max,
                      p_max=p_max, delay_table=delay_table)
    print("Creating transition tensor")
    T = transition_tensor(state_space=states, action_space=actions, services=S, nodes=N, edges=E, p_max=p_max,
                          c_max=c_max)
    print("Starting Value Iteration")
    V = value_iteration(T=T, num_states=len(states), num_actions=len(actions), R=R, theta=theta, gamma=gamma)
    print(f"Value function: {V}")
    pi = greedy_policy(T=T, R=R, gamma=gamma, V=V, num_states=len(states), num_actions=len(actions))
    print(f"Greedy policy: {pi}")
    print("Saving value function to disk")
    with open('V.npy', 'wb') as f:
        np.save(f, np.asarray(V))
    print("Saving policy to disk")
    with open('pi.npy', 'wb') as f:
        np.save(f, np.asarray(pi))


def run_pi(states: List, actions: List, gamma: float, S: list, N: list, E: list, p_max: float, c_max: float,
           num_iterations: int, eval_every: int, eval_batch_size: int,
           running_average_size, eval_initial_state: int, eval_horizon: int, theta: float, delay_table: List) -> None:
    """
    Runs a given base policy

    :param states: the state space
    :param actions: the action space
    :param gamma: the discount factor
    :param S: the set of services
    :param N: the set of nodes
    :param E: the set of edges
    :param p_max: the maximum value for the routing action
    :param c_max: the maximum value for the scaling action
    :param num_iterations: the number of iterations to run
    :param eval_every: frequency of evaluation
    :param eval_batch_size: the batch size for evaluation
    :param running_average_size: the running average size for evaluation
    :param eval_initial_state: the initial state for evaluation
    :param eval_horizon: the horizon for the evaluation
    :param theta: the theta to measure convergence of value iteration
    :param delay_table: the table with the predicted response times of different services in different states
    :return: None
    """
    print("Creating reward tensor")
    R = reward_tensor(state_space=states, action_space=actions, services=S, nodes=N, edges=E, c_max=c_max, p_max=p_max,
                      delay_table=delay_table)
    print("Creating transition tensor")
    T = transition_tensor(state_space=states, action_space=actions, services=S, nodes=N, edges=E, p_max=p_max,
                          c_max=c_max)
    pi, V, average_returns, running_average_returns = policy_iteration(
        T=T, N=num_iterations, gamma=gamma, R=R,
        num_states=len(states), num_actions=len(actions), eval_every=eval_every,
        eval_batch_size=eval_batch_size, eval_initial_state=eval_initial_state, eval_horizon=eval_horizon,
        running_average_size=running_average_size, theta=theta)
    print(f"Policy: {pi}")
    print(f"Value function: {V}")
    print("Saving value function to disk")
    with open('V.npy', 'wb') as f:
        np.save(f, np.asarray(V))
    print("Saving policy to disk")
    with open('pi.npy', 'wb') as f:
        np.save(f, np.asarray(pi))


def compute_value_function_of_policy(base_pi: Union[List, MaskablePPO], filename: str, batch_size: int, states: List,
                                     actions: List, gamma: float, S: list, N: list, E: list, p_max: float,
                                     delta_c: float, delta_p: float,
                                     c_max: float, delay_table: List, env=None) -> None:
    """
    Estimates the value function of a given base policy

    :param base_pi: the base policy
    :param filename: the filename to save the value function
    :param batch_size: the batch size for Monte-Carlo estimation
    :param states: the state space
    :param actions: the action space
    :param gamma: the discount factor
    :param S: the set of services
    :param N: the set of nodes
    :param E: the set of edges
    :param p_max: the maximum routing action value
    :param delta_c: the discretization step of the scaling action
    :param delta_p: the discretization step of the routing action
    :param c_max: the maximum value of the scaling action
    :param delay_table: the table with the predicted response times
    :param env: the RL environment
    :return: None
    """
    V_base = monte_carlo_policy_evaluation(policy=base_pi, states=states, actions=actions, gamma=gamma,
                                           batch_size=batch_size, N=N, E=E, c_max=c_max, p_max=p_max, S=S,
                                           delay_table=delay_table, delta_c=delta_c,
                                           delta_p=delta_p, env=env)
    V_base = np.array(V_base)
    with open(filename, 'wb') as f:
        np.save(f, np.asarray(V_base))


def compute_delay_table(states: List, S: List, filename: str) -> None:
    """
    Computes the table with the predicted response times of different services in different states
    :param states: the state space
    :param S: the set of services
    :param filename: the filename to save the table
    :return: None
    """
    delay_table = precompute_delays(states=states, S=S)
    delay_table = np.array(delay_table)
    with open(filename, 'wb') as f:
        np.save(f, np.asarray(delay_table))


def greedy_base_policy(states: List, actions: List, S: List, N: List, E: List, c_max: float, p_max: float,
                       delay_table: List, filename: str) -> None:
    """
    Computes and saves the greedy base policy

    :param states: the state space
    :param actions: the action space
    :param S: the set of services
    :param N: the set of nodes
    :param E: the set of edges
    :param c_max: the maximum scaling action value
    :param p_max: the maximum routing action value
    :param delay_table: a table with the predicted response times
    :param filename: the filename to save the policy
    :return: None
    """
    policy = np.zeros([len(states), len(actions)])
    for i, s in enumerate(states):
        a_rews = []
        for j, a in enumerate(actions):
            a_cost = sum(list(map(lambda x: abs(x), a)))
            a_rews.append(reward_function(s=s, a=a, services=S, nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                          states=states, delay_table=delay_table) - a_cost * 0.001)
        a_rews = np.array(a_rews)
        greedy_a = np.argmax(a_rews)
        policy[i][greedy_a] = 1

    with open(filename, 'wb') as f:
        np.save(f, np.asarray(policy))


def optimal_base_policy(states: List, actions: List, filename: str) -> None:
    """
    Computes and stores the optimal base policy (upper bound)

    :param states: the state space
    :param actions: the action space
    :param filename: the filename to save the policy
    :return: None
    """
    policy = np.zeros([len(states), len(actions)])
    for i, s in enumerate(states):
        if s == [2, 2, 1, 1]:
            a = actions.index(tuple([0, 1, -0.2, -0.2]))
            policy[i][a] = 1
        elif s == [2, 3, 0.8, 0.8]:
            a = actions.index(tuple([0, 1, -0.2, -0.2]))
            policy[i][a] = 1
        elif s == [2, 4, 0.6, 0.6]:
            a = actions.index(tuple([1, 1, -0.2, -0.2]))
            policy[i][a] = 1
        elif s == [3, 5, 0.4, 0.4]:
            a = actions.index(tuple([0, 0, -0.2, -0.2]))
            policy[i][a] = 1
        elif s == [3, 5, 0.2, 0.2]:
            a = actions.index(tuple([0, 0, -0.2, -0.2]))
            policy[i][a] = 1
        else:
            a = actions.index(tuple([0, 0, 0, 0]))
            policy[i][a] = 1

    with open(filename, 'wb') as f:
        np.save(f, np.asarray(policy))


def random_base_policy(states: List, actions: List) -> npt.NDArray[Any]:
    """
    Gets the tabular random base policy

    :param states: the state space
    :param actions: the action space
    :return: a table defining the random base policy
    """
    return np.ones([len(states), len(actions)]) / len(actions)


if __name__ == '__main__':
    S = services()
    N = nodes()
    E = edges()

    # DP parameters
    # delta_c = 1
    # c_max = 2
    # delta_p = 0.5
    # p_max = 1
    # theta = 0.05
    # gamma = 0.9

    # MC parameters
    delta_c = 1
    c_max = 5
    delta_p = 0.2
    p_max = 1
    gamma = 0.99
    theta = 0.01

    # Setup spaces
    print("Creating state space")
    states = state_space(services=S, nodes=N, edges=E, delta_c=delta_c, delta_p=delta_p, p_max=p_max, c_max=c_max)
    print("Creating action space")
    actions, component_spaces = action_space(services=S, nodes=N, edges=E, delta_c=delta_c, delta_p=delta_p)

    # Compute delay table
    compute_delay_table(states=states, S=S, filename="delay_table.npy")
    print("Loading delay table")
    with open('delay_table.npy', 'rb') as f:
        delay_table = list(np.load(f, allow_pickle=True).tolist())

    # Compute random policy
    random_policy = random_base_policy(states=states, actions=actions)

    # Compute greedy policy
    print("computing greedy policy")
    greedy_base_policy(states=states, actions=actions, S=S, N=N, E=E, c_max=c_max, p_max=p_max, delay_table=delay_table,
                       filename="pi_greedy.npy")
    print("loading greedy policy")
    with open('pi_greedy.npy', 'rb') as f:
        pi_greedy = list(np.load(f, allow_pickle=True).tolist())
    print("greedy policy loaded")

    # Compute optimal policy
    print("computing optimal policy")
    optimal_base_policy(states=states, actions=actions, filename="pi_opt.npy")
    print("loading optimal policy")
    with open('pi_opt.npy', 'rb') as f:
        pi_opt = list(np.load(f, allow_pickle=True).tolist())
    print("optimal policy loaded")

    print("loading PPO policy")
    env = RoutingEnv()
    try:
        pi_ppo = MaskablePPO.load("/Users/kimham/workspace/NOMS2024_rollouts/sc1/training/self_routing_14.zip")
    except:
        try:
            pi_ppo = MaskablePPO.load("/Users/kim/workspace/NOMS2024_rollouts/sc1/training/self_routing_14.zip")
        except:
            pi_ppo = MaskablePPO.load("/home/kim/workspace/NOMS2024_rollouts/sc1/training/self_routing_14.zip")
    print("PPO policy loaded")

    initial_state = [2.0, 2.0, 1.0, 1.0]
    initial_state_index = states.index(initial_state)
    print(f"initial state: {initial_state_index}")

    # VI
    run_vi(states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max, c_max=c_max, theta=theta,
           delay_table=delay_table)

    # PI
    run_pi(states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max, c_max=c_max, eval_every=1,
           eval_batch_size=10, running_average_size=1, eval_initial_state=len(states) - 1, eval_horizon=10,
           num_iterations=100, theta=theta, delay_table=delay_table)

    # Exact online play
    base_policy = random_policy
    run_exact_one_step_online_play(base_pi=base_policy, states=states, actions=actions, gamma=gamma, theta=theta,
                                   S=S, N=N,
                                   E=E, p_max=p_max, c_max=c_max, initial_state=41, horizon=10)

    # Compute base policy value function

    print("Computing value function of random base policy")
    compute_value_function_of_policy(base_pi=random_policy, filename="random_base_policy_V.npy", batch_size=1,
                                     states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max,
                                     c_max=c_max, delay_table=delay_table, delta_c=delta_c, delta_p=delta_p)

    print("Computing value function of optimal base policy")
    compute_value_function_of_policy(base_pi=pi_opt, filename="opt_base_policy_V.npy", batch_size=1,
                                     states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max,
                                     c_max=c_max, delay_table=delay_table, delta_c=delta_c, delta_p=delta_p)

    print("Computing value function of greedy base policy")
    compute_value_function_of_policy(base_pi=pi_greedy, filename="greedy_base_policy_V.npy", batch_size=1,
                                     states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max,
                                     c_max=c_max, delay_table=delay_table, delta_c=delta_c, delta_p=delta_p)
    print("Computing value function of PPO base policy")
    compute_value_function_of_policy(base_pi=pi_ppo, filename="ppo_base_policy_V.npy", batch_size=1,
                                     states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max,
                                     c_max=c_max, delay_table=delay_table, delta_p=delta_p,
                                     delta_c=delta_c, env=env)

    # Monte Carlo online play

    seeds = [151, 125, 1251, 96, 215, 21, 5678, 2, 3256, 15]

    base_policy = random_policy
    with open('random_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    means, stds, time_means, time_stds = run_monte_carlo_one_step_online_play(base_pi=base_policy, states=states,
                                                                              actions=actions, gamma=gamma,
                                                                              S=S, N=N, E=E, p_max=p_max, c_max=c_max,
                                                                              initial_state=initial_state_index,
                                                                              horizon=20, batch_size=5,
                                                                              V_base=list(V_base.tolist()), l=1,
                                                                              deterministic_model=True,
                                                                              delay_table=delay_table,
                                                                              num_episodes=5,
                                                                              seeds=seeds, env=env, multiagent=False,
                                                                              component_spaces=None)
    with open('random_rollout_reward_means.npy', 'wb') as f:
        np.save(f, np.array(means))
    with open('random_rollout_reward_stds.npy', 'wb') as f:
        np.save(f, np.array(stds))

    base_policy = pi_greedy
    with open('greedy_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    means, stds, time_means, time_stds = run_monte_carlo_one_step_online_play(base_pi=base_policy, states=states,
                                                                              actions=actions, gamma=gamma,
                                                                              S=S, N=N, E=E, p_max=p_max, c_max=c_max,
                                                                              initial_state=initial_state_index,
                                                                              horizon=20, batch_size=5,
                                                                              V_base=list(V_base.tolist()), l=1,
                                                                              deterministic_model=True,
                                                                              delay_table=delay_table,
                                                                              num_episodes=5, seeds=seeds, env=env,
                                                                              multiagent=False,
                                                                              component_spaces=None)
    with open('greedy_rollout_reward_means.npy', 'wb') as f:
        np.save(f, np.array(means))
    with open('greedy_rollout_reward_stds.npy', 'wb') as f:
        np.save(f, np.array(stds))

    with open('ppo_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    base_policy = pi_ppo
    means, stds, time_means, time_stds = run_monte_carlo_one_step_online_play(base_pi=base_policy, states=states,
                                                                              actions=actions, gamma=gamma,
                                                                              S=S, N=N, E=E, p_max=p_max, c_max=c_max,
                                                                              initial_state=initial_state_index,
                                                                              horizon=20, batch_size=5,
                                                                              V_base=list(V_base.tolist()), l=1,
                                                                              deterministic_model=True,
                                                                              delay_table=delay_table,
                                                                              delta_c=delta_c, delta_p=delta_p,
                                                                              num_episodes=5, seeds=seeds,
                                                                              env=env, multiagent=False,
                                                                              component_spaces=None)
    with open('ppo_rollout_reward_means.npy', 'wb') as f:
        np.save(f, np.array(means))
    with open('ppo_rollout_reward_stds.npy', 'wb') as f:
        np.save(f, np.array(stds))

    # Evaluate base policies
    base_policy = random_policy
    with open('random_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    means, stds = run_base_policy_online(base_pi=base_policy, states=states, actions=actions,
                                         S=S, N=N, E=E, p_max=p_max, c_max=c_max,
                                         initial_state=initial_state_index,
                                         horizon=20, V_base=list(V_base.tolist()),
                                         delay_table=delay_table,
                                         num_episodes=5, seeds=seeds, delta_c=delta_c,
                                         delta_p=delta_p, env=env)
    with open('random_base_reward_means.npy', 'wb') as f:
        np.save(f, np.array(means))
    with open('random_base_reward_stds.npy', 'wb') as f:
        np.save(f, np.array(stds))

    base_policy = pi_greedy
    with open('greedy_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    means, stds = run_base_policy_online(base_pi=base_policy, states=states, actions=actions,
                                         S=S, N=N, E=E, p_max=p_max, c_max=c_max,
                                         initial_state=initial_state_index,
                                         horizon=20, V_base=list(V_base.tolist()),
                                         delay_table=delay_table,
                                         num_episodes=5, seeds=seeds, delta_c=delta_c,
                                         delta_p=delta_p,
                                         env=env)
    with open('greedy_base_reward_means.npy', 'wb') as f:
        np.save(f, np.array(means))
    with open('greedy_base_reward_stds.npy', 'wb') as f:
        np.save(f, np.array(stds))

    base_policy = pi_ppo
    with open('ppo_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    means, stds = run_base_policy_online(base_pi=base_policy, states=states, actions=actions,
                                         S=S, N=N, E=E, p_max=p_max, c_max=c_max,
                                         initial_state=initial_state_index,
                                         horizon=20, V_base=list(V_base.tolist()),
                                         delay_table=delay_table,
                                         num_episodes=5, seeds=seeds, delta_c=delta_c,
                                         delta_p=delta_p, env=env)
    # print(means)
    with open('ppo_base_reward_means.npy', 'wb') as f:
        np.save(f, np.array(means))
    with open('ppo_base_reward_stds.npy', 'wb') as f:
        np.save(f, np.array(stds))

    base_policy = pi_opt
    with open('opt_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    means, stds = run_base_policy_online(base_pi=base_policy, states=states, actions=actions,
                                         S=S, N=N, E=E, p_max=p_max, c_max=c_max,
                                         initial_state=initial_state_index,
                                         horizon=20, V_base=list(V_base.tolist()),
                                         delay_table=delay_table,
                                         num_episodes=5, seeds=seeds, delta_c=delta_c,
                                         delta_p=delta_p, env=env)
    with open('opt_base_reward_means.npy', 'wb') as f:
        np.save(f, np.array(means))
    with open('opt_base_reward_stds.npy', 'wb') as f:
        np.save(f, np.array(stds))

    # Newton step
    print("Start Newton")

    base_policy = random_policy
    alt_policy = pi_greedy
    with open('random_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    with open('greedy_base_policy_V.npy', 'rb') as f:
        V_alt = np.load(f, allow_pickle=True)
    V_rollout_1, V_rollout_2, V_base, V_base_2, T_rollout_slope, T_rollout_intercept, T_base_slope, T_base_intercept = \
        newton_step_data(base_pi=base_policy, states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max,
                         c_max=c_max, initial_state=initial_state_index, batch_size=5,
                         V_base=list(V_base.tolist()), l=1, deterministic_model=True, delay_table=delay_table,
                         alt_pi=alt_policy, alt_V=V_alt, env=env)

    print(f"V_rollout_1: {V_rollout_1}, V_rollout_2: {V_rollout_2}, V_base_1: {V_base}, V_base_2: {V_base_2}, "
          f"T_rollout_slope: {T_rollout_slope}, T_rollout_intercept: {T_rollout_intercept}, "
          f"T_base_slope: {T_base_slope}, T_base_intercept: {T_base_intercept}")

    base_policy = pi_greedy
    alt_policy = random_policy
    with open('greedy_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    with open('random_base_policy_V.npy', 'rb') as f:
        V_alt = np.load(f, allow_pickle=True)
    V_rollout_1, V_rollout_2, V_base, V_base_2, T_rollout_slope, T_rollout_intercept, T_base_slope, T_base_intercept = \
        newton_step_data(base_pi=base_policy, states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max,
                         c_max=c_max, initial_state=initial_state_index, batch_size=5,
                         V_base=list(V_base.tolist()), l=1, deterministic_model=True, delay_table=delay_table,
                         alt_pi=alt_policy, alt_V=V_alt, env=env)

    print(f"V_rollout_1: {V_rollout_1}, V_rollout_2: {V_rollout_2}, V_base_1: {V_base}, V_base_2: {V_base_2}, "
          f"T_rollout_slope: {T_rollout_slope}, T_rollout_intercept: {T_rollout_intercept}, "
          f"T_base_slope: {T_base_slope}, T_base_intercept: {T_base_intercept}")

    base_policy = pi_ppo
    alt_policy = pi_greedy
    with open('ppo_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    with open('greedy_base_policy_V.npy', 'rb') as f:
        V_alt = np.load(f, allow_pickle=True)
    V_rollout_1, V_rollout_2, V_base, V_base_2, T_rollout_slope, T_rollout_intercept, T_base_slope, T_base_intercept = \
        newton_step_data(base_pi=base_policy, states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max,
                         c_max=c_max, initial_state=initial_state_index, batch_size=5,
                         V_base=list(V_base.tolist()), l=1, deterministic_model=True, delay_table=delay_table,
                         alt_pi=alt_policy, alt_V=V_alt, env=env)

    print(f"V_rollout_1: {V_rollout_1}, V_rollout_2: {V_rollout_2}, V_base_1: {V_base}, V_base_2: {V_base_2}, "
          f"T_rollout_slope: {T_rollout_slope}, T_rollout_intercept: {T_rollout_intercept}, "
          f"T_base_slope: {T_base_slope}, T_base_intercept: {T_base_intercept}")

    # Ablation study for lookahead horizon l

    seeds = [151, 125, 1251, 96, 215, 21, 5678, 2, 3256, 15]
    multiagent = True
    base_policy = random_policy
    with open('random_base_policy_V.npy', 'rb') as f:
        V_base = np.load(f, allow_pickle=True)
    for l in range(1, 5):
        means, stds, time_means, time_stds = run_monte_carlo_one_step_online_play(
            base_pi=base_policy, states=states, actions=actions, gamma=gamma, S=S, N=N, E=E, p_max=p_max, c_max=c_max,
            initial_state=initial_state_index, horizon=20, batch_size=5, V_base=list(V_base.tolist()), l=l,
            deterministic_model=True, delay_table=delay_table, num_episodes=1, seeds=seeds, env=env,
            multiagent=multiagent, component_spaces=component_spaces)
        with open(f'random_rollout_reward_means_{l}.npy', 'wb') as f:
            np.save(f, np.array(means))
        with open(f'random_rollout_reward_stds_{l}.npy', 'wb') as f:
            np.save(f, np.array(stds))
        with open(f'random_rollout_time_means_{l}.npy', 'wb') as f:
            np.save(f, np.array(time_means))
        with open(f'random_rollout_time_stds_{l}.npy', 'wb') as f:
            np.save(f, np.array(time_stds))
