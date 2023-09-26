"""
Implements the Policy Iteration algorithm
"""
from typing import List, Tuple
import numpy as np


def policy_evaluation(policy: List, num_states: int, gamma: float, theta: float, T, R) -> List:
    """
    The policy evaluation function

    :param policy: the policy to evaluate
    :param num_states: the number of states
    :param gamma: the discount factor
    :param theta: the theta value for measuring convergence
    :param T: the transition tensor
    :param R: the reward tensor
    :return: the value function of the policy
    """
    V = np.zeros(num_states)
    while True:
        delta = 0
        for s in range(num_states):
            v = 0
            for s_prime in range(num_states):
                for a, action_prob in enumerate(policy[s]):
                    transition_prob = T[s][s_prime][a]
                    r = R[s][a]
                    v += action_prob * transition_prob * (r + gamma * V[s_prime])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return list(V.tolist())


def one_step_lookahead(s: int, V: List, gamma: float, num_actions: int, num_states: int, T: List, R: List) -> List:
    """
    Performs one-step lookahead maximization

    :param s: the current state
    :param V: the value function
    :param gamma: the discount factor
    :param num_actions: the number of actions
    :param num_states: the number of states
    :param T: the transition tensor
    :param R: the reward tensor
    :return: the action values
    """
    A = np.zeros(num_actions)
    for a in range(num_actions):
        for s_prime in range(num_states):
            transition_prob = T[s][s_prime][a]
            r = R[s][a]
            A[a] += transition_prob * (r + gamma * V[s_prime])
    return A


def policy_improvement(V_pi: List, gamma: float, num_states: int, policy: List, num_actions: int, T: List, R: List) \
        -> Tuple[List, bool, int]:
    """
    The policy improvement functin

    :param V_pi: the value function of the base policy
    :param gamma: the discount factor
    :param num_states: the number of states
    :param policy: the base policy
    :param num_actions: the number of actions
    :param T: the transition tensor
    :param R: the reward tensor
    :return: the policy, a boolean flag indiciating whether the policy has convergend, and the number of changed states
    """
    num_changed_states = 0
    policy_stable = True
    for s in range(num_states):
        chosen_a = np.argmax(policy[s])
        action_values = one_step_lookahead(s, V_pi, gamma=gamma, num_actions=num_actions, num_states=num_states,
                                           T=T, R=R)
        best_a = np.argmax(action_values)
        if chosen_a != best_a:
            policy_stable = False
            num_changed_states += 1
        policy[s] = np.eye(num_actions)[best_a]
    return policy, policy_stable, num_changed_states


def policy_iteration(num_states: int, num_actions: int, T: List, R: List, N: int, gamma: float,
                     eval_every: int, eval_batch_size: int, running_average_size,
                     eval_initial_state: int, eval_horizon: int, theta: float) -> Tuple[List, List, List, List]:
    """
    Implements the policy iteration algorithm

    :param num_states: the number of states
    :param num_actions: the number of actions
    :param T: the transition tensor
    :param R: the reward tensor
    :param N: the number of nodes
    :param gamma: the discount factor
    :param eval_every: the evaluation frequency
    :param eval_batch_size: the batch size for evaluation
    :param running_average_size: the number of measurements to compute a running average
    :param eval_initial_state: the initial state for evaluation
    :param eval_horizon: the horizon for evaluation
    :param theta: the theta perameter to measure convergence
    :return: the optimal policy, its value function, the average returns of the policy, and the running averages.
    """
    print("Staring policy iteration")
    policy = np.ones([num_states, num_actions]) / num_actions
    average_returns = []
    running_average_returns = []
    for i in range(0, N):
        # Evaluate the current policy
        V_pi = policy_evaluation(policy=policy, gamma=gamma, num_states=num_states, theta=theta, T=T, R=R)

        # Will be set to false if we make any changes to the policy
        policy, policy_stable, num_changed_states = policy_improvement(V_pi=V_pi, gamma=gamma, num_states=num_states,
                                                                       num_actions=num_actions, policy=policy, T=T, R=R)

        if i % eval_every == 0:
            avg_return = evaluate_policy(T=T, initial_state=eval_initial_state, policy=policy,
                                         eval_batch_size=eval_batch_size, horizon=eval_horizon, R=R)
            average_returns.append(avg_return)
            running_avg_J = running_average(average_returns, running_average_size)
            running_average_returns.append(running_avg_J)
            print(f"PI, iteration: {i}, stable policy: {policy_stable}, changes: {num_changed_states}, "
                  f"average return: {avg_return}")
        if policy_stable:
            print("PI converged")
            return policy, V_pi, average_returns, running_average_returns


def running_average(x: List, N: int) -> List:
    """
    Utility function to compute a running average of a list

    :param x: the list
    :param N: the number of elements to include in the running average
    :return: the running averaged list
    """
    if len(x) < N:
        N = len(x)
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return round(y[-1], 2)


def evaluate_policy(T: List, initial_state: int, eval_batch_size: int, horizon: int, policy: List,
                    R: List) -> float:
    """
    Utility function for evaluating a given policy

    :param T: the transition tensor
    :param initial_state: the initial state
    :param eval_batch_size: the batch size for evaluation
    :param horizon: the horizon for evaluation
    :param policy: the policy to evaluate
    :param R: the reward tensor
    :return: the average return of teh policy
    """
    T_1 = np.transpose(np.array(T), (2, 0, 1))
    returns = []
    for i in range(eval_batch_size):
        s = initial_state
        ret = 0
        for t in range(horizon):
            a = np.random.choice(np.arange(0, len(policy[s])), p=policy[s])
            s_prime = np.random.choice(np.arange(0, len(T_1[a, s, :])), p=T_1[a, s, :])
            r = R[s][a]
            s = s_prime
            ret += r
        returns.append(ret)
    avg_return = np.mean(returns)
    return float(avg_return)
