"""
Defines the system model for scenario 1
"""
from typing import List, Tuple, Any
import numpy as np
import numpy.typing as npt
import itertools
import joblib

# Environment constants
l_1 = 4.0
c_l_1 = 4.0
l_2 = 15.0
c_l_2 = 15.0
O_1 = 0.95
O_2 = 0.115
print("Loading delay model from disk")
try:
    delay_model = joblib.load("/Users/kimham/Dropbox/NOMS24/SC1/delays_RF_model.joblib")
except:
    try:
        delay_model = joblib.load("/Users/kim/Dropbox/NOMS24/SC1/delays_RF_model.joblib")
    except:
        delay_model = joblib.load("/home/kim/Dropbox/NOMS24/SC1/delays_RF_model.joblib")
print("Delay model loaded successfully")


def services() -> List[int]:
    """
    :return: the set of services
    """
    return [1, 2]


def nodes() -> List[int]:
    """
    :return: the set of nodes
    """
    return [0, 1]


def edges() -> List[Tuple[int, int]]:
    """
    :return: the set of edges
    """
    return [(0, 1)]


def precompute_delays(states: List, S: List) -> npt.NDArray[Any]:
    """
    Function for precomputing the delay table

    :param states: the state space
    :param S: the list of services
    :return: the delay table
    """
    delay_table = np.zeros((len(states), len(S)))
    for i, s in enumerate(states):
        print(f"computing delay for state {i}/{len(states)}")
        [[d_1, d_2]] = delay_model.predict([[s[0] * 1000, s[1] * 1000, s[2], s[3], l_1, c_l_1, l_2, c_l_2]])
        delay_table[i] = [d_1, d_2]
    return delay_table


def state_space(services: List[int], nodes: List[int], edges: List[Tuple[int, int]], delta_c: float,
                delta_p: float, c_max: float, p_max: float) -> List:
    """
    Gets the state space of the model

    :param services: the set of services
    :param nodes: the set of nodes
    :param edges: the set of edges
    :param delta_c: the discretization step of the scaling action
    :param delta_p: the discretization step of the routing action
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :return: The state space
    """
    routing_weights = list(
        map(lambda x: round(x, 3), list(np.linspace(0, p_max, num=int((p_max / delta_p)) + 1).tolist())))
    cpu_allocations = list(map(lambda x: round(x, 3), list(np.linspace(1, c_max, num=int((c_max / delta_c))).tolist())))
    component_spaces = []
    for _ in nodes:
        component_spaces.append(cpu_allocations)
    for _ in services:
        for _ in edges:
            component_spaces.append(routing_weights)
    state_space = [list(t) for t in list(itertools.product(*component_spaces))]
    return state_space


def action_space(services: List[int], nodes: List[int], edges: List[Tuple[int, int]], delta_c: float, delta_p: float) \
        -> List:
    """
    Computes the action space of the model

    :param services: the set of services
    :param nodes: the set of nodes
    :param edges: the set of edges
    :param delta_c: the discretization step of the scaling action
    :param delta_p: the discretization step of the routing action
    :return: the action space
    """
    routing_increments = [-delta_p, 0, delta_p]
    cpu_increments = [-delta_c, 0, delta_c]
    component_spaces = []
    for _ in nodes:
        component_spaces.append(cpu_increments)
    for _ in services:
        for _ in edges:
            component_spaces.append(routing_increments)
    action_space = list(itertools.product(*component_spaces))
    return action_space, component_spaces


def transition_tensor(state_space: List, action_space: List, c_max: float, p_max: float, services: List[int],
                      nodes: List[int], edges: List[Tuple[int, int]]) -> List:
    """
    Gets the transition tensor of the system model

    :param state_space: the state space
    :param action_space: the action space
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :param services: the set of services
    :param nodes: the set of nodes
    :param edges: the set of edges
    :return: The transition tensor
    """
    T = []
    for i, s in enumerate(state_space):
        print(f"Generating transition tensor, {i}/{len(state_space) - 1}")
        s_transitions = []
        for s_prime in state_space:
            s_prime_transitions = []
            for a in action_space:
                true_s_prime = transition_function(s=s, a=a, services=services, nodes=nodes, edges=edges, c_max=c_max,
                                                   p_max=p_max)
                if list(s_prime) == list(true_s_prime):
                    s_prime_transitions.append(1)
                else:
                    s_prime_transitions.append(0)
            s_transitions.append(s_prime_transitions)
        T.append(s_transitions)
    return T


def reward_function(s: List, a: List, services: List[int], nodes: List[int], edges: List[Tuple[int, int]],
                    c_max: float, p_max: float, states: List, delay_table=None) -> float:
    """
    The reward function of the model

    :param s: the state
    :param a: the action
    :param services: the set of services
    :param nodes: the set of nodes
    :param edges: the set of edges
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :param states: the state space
    :param delay_table: the table with the precomputed delays for different services in different states
    :return: the reward
    """
    s_prime = transition_function(s=s, a=a, services=services, nodes=nodes, edges=edges, c_max=c_max, p_max=p_max)
    if delay_table is None:
        [[d_1, d_2]] = delay_model.predict([[s_prime[0] * 1000, s_prime[1] * 1000, s_prime[2], s_prime[3],
                                             l_1, c_l_1, l_2, c_l_2]])
    else:
        [d_1, d_2] = delay_table[states.index(s_prime)]
    if (d_1 <= O_1) and (d_2 <= O_2):
        overall_cpu = (s[0] + s[1]) / 10
        reward = (1 - overall_cpu) * (1 - s[2]) * (1 - s[3])
    else:
        reward = 0
        if d_1 > O_1:
            reward -= d_1 - O_1
        if d_2 > O_2:
            reward -= d_2 - O_2
    return reward


def reward_tensor(state_space: List, action_space: List, services: List[int], nodes: List[int],
                  edges: List[Tuple[int, int]], c_max: float, p_max: float, delay_table: List) -> List:
    """
    Computes the reward tensor

    :param state_space: the state space
    :param action_space: the action space
    :param services: the set of services
    :param nodes: the set of nodes
    :param edges: the set of edges
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :param delay_table: the table with the precomputed delays
    :return: the reward tensor
    """
    R = []
    for s in state_space:
        rewards = []
        for a in action_space:
            rewards.append(reward_function(s=s, a=a, services=services, nodes=nodes, edges=edges, c_max=c_max,
                                           p_max=p_max, states=state_space, delay_table=delay_table))
        R.append(rewards)
    return R


def transition_function(s: List, a: List, services: List[int], nodes: List[int], edges: List[Tuple[int, int]],
                        c_max: float, p_max: float) -> List:
    """
    The transition function of the model

    :param s: the current state
    :param a: the current action
    :param services: the set of services
    :param nodes: the set of nodes
    :param edges: the set of edges
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :return: the next state
    """
    s_prime = []
    for i in range(len(nodes)):
        s_prime.append(round(max(1.0, min(c_max, s[i] + a[i])), 2))
    for i in range(len(nodes), len(nodes) + len(services) * len(edges)):
        s_prime.append(round(max(0.0, min(p_max, s[i] + a[i])), 2))
    return s_prime


def convert_dp_s_to_env_s(s: List) -> npt.NDArray[Any]:
    """
    Utility function for converting between two state representations

    :param s: convert DP state to RL state
    :return: the RL state representation
    """
    return np.array([s[2], s[3], s[0] * 1000, s[1] * 1000])


def convert_env_a_to_dp_a(a: List, delta_c: float, delta_p: float) -> List:
    """
    Utility function for converting between two action representations

    :param a: the RL action representation
    :param delta_c: the scaling action discretization step
    :param delta_p: the routing action discretization step
    :return: the DP action representation
    """
    converted = [a[2], a[3], a[0], a[1]]

    if converted[0] == 1:
        converted[0] = -delta_c
    elif converted[0] == 2:
        converted[0] = delta_c
    elif converted[0] == 0:
        converted[0] = 0

    if converted[1] == 1:
        converted[1] = -delta_c
    elif converted[1] == 2:
        converted[1] = delta_c
    elif converted[1] == 0:
        converted[1] = 0

    if converted[2] == 1:
        converted[2] = -delta_p
    elif converted[2] == 2:
        converted[2] = delta_p
    elif converted[2] == 0:
        converted[2] = 0

    if converted[3] == 1:
        converted[3] = -delta_p
    elif converted[3] == 2:
        converted[3] = delta_p
    elif converted[3] == 0:
        converted[3] = 0
    return converted
