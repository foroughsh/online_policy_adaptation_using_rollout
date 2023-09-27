"""
Implements the value iteration algorithm for scenario 1
"""
from typing import List, Tuple
import numpy as np


def one_step_lookahead_maximization(s, V, num_actions, num_states, T, gamma, R) -> Tuple[int, float]:
    """
    One-step lookahead maximization for the value iteration algorithm

    :param s: the current state
    :param V: the value function
    :param num_actions: the number of actions
    :param num_states: the number of states
    :param T: the transition tensor
    :param gamma: the discount factor
    :param R: the reward tensor
    :return: the best action and its value
    """
    A = np.zeros(num_actions)
    for a in range(num_actions):
        reward = R[s][a]
        for next_state in range(num_states):
            prob = T[s][next_state][a]
            A[a] += prob * (reward + gamma * V[next_state])
    best_action = np.argmax(A)
    best_action_value = A[best_action]
    return int(best_action), float(best_action_value)


def value_iteration(T: List, num_states: int, num_actions: int, R: List, theta=0.0001, gamma=1.0) -> List:
    """
    The value iteration algorithm

    :param T: the transition tensor
    :param num_states: the number of states
    :param num_actions: the number of actions
    :param R: the reward tensor
    :param theta: the theta parameter to measure convergence
    :param gamma: the discount factor
    :return: the value function
    """
    V = np.zeros(num_states)
    iteration = 0
    while True:
        delta = 0
        for s in range(num_states):
            best_action, best_action_value = one_step_lookahead_maximization(s=s, V=V, num_actions=num_actions,
                                                                             num_states=num_states, T=T, gamma=gamma,
                                                                             R=R)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        iteration += 1
        print(f"VI iteration: {iteration}, delta: {delta}, theta: {theta}")
        if delta < theta:
            break
    print("VI converged")
    return V.tolist()


def greedy_policy(T: List, R: List, gamma: float, V: List, num_states: int, num_actions: int) -> List:
    """
    Computes the greedy policy based on a given value function

    :param T: the transition tensor
    :param R: the reward tensor
    :param gamma: the discount factor
    :param V: the value function
    :param num_states: the number of states
    :param num_actions: the number of actions
    :return: the greedy policy
    """
    pi_prime = np.zeros((num_states, num_actions))
    for s in range(0, num_states):
        action_values = np.zeros(num_actions)
        for a in range(0, num_actions):
            action_values[a] = R[s][a]
        best_action = np.argmax(action_values)
        pi_prime[s, best_action] = 1
    return pi_prime
