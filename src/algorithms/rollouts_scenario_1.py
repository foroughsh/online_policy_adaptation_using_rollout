"""
Script containing rollout algorithms for scenario 1
"""
from typing import List, Union, Tuple
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from policy_iteration_scenario_1 import policy_evaluation
from system_model.system_model_scenario_1 import (
    transition_function, reward_function, convert_dp_s_to_env_s, convert_env_a_to_dp_a)


def exact_one_step_rollout_policy(base_pi: List, V_base, s: int, gamma: float, num_actions: int, num_states: int,
                                  T: List, R: List, theta: float) -> Tuple[List, List]:
    """
    Computes the exact one-step lookahead rollout policy

    :param base_pi: the base policy
    :param V_base: value function of the base policy
    :param s: the current state
    :param gamma: the discount factor
    :param num_actions: the number of actions
    :param num_states: the number of states
    :param T: the transition tensor
    :param R: the reward tensor
    :param theta: the theta parameter for vale iteration convergence
    :return:
    """
    a = exact_one_step_lookahead_maximization(s=s, V=V_base, gamma=gamma, num_actions=num_actions,
                                              num_states=num_states, T=T, R=R)
    rollout_pi = base_pi.copy()
    rollout_pi[s] = np.zeros(len(base_pi[s]))
    rollout_pi[s][a] = 1
    V_rollout = policy_evaluation(policy=rollout_pi, num_states=num_states, gamma=gamma, theta=theta, T=T, R=R)
    return rollout_pi, V_rollout


def exact_one_step_lookahead_maximization(s: int, V: List, gamma: float, num_actions: int, num_states: int, T: List,
                                          R: List) -> int:
    """
    Gets the best action from one-step lookahead maximization

    :param s: the current state
    :param V: the value function
    :param gamma: the discount factor
    :param num_actions: the number of actions
    :param num_states: the number of states
    :param T: the transition tensor
    :param R: the reward tensor
    :return: the best action
    """
    A = np.zeros(num_actions)
    for a in range(num_actions):
        for s_prime in range(num_states):
            transition_prob = T[s][s_prime][a]
            r = R[s][a]
            A[a] += transition_prob * (r + gamma * V[s_prime])
    best_a = np.argmax(A)
    return int(best_a)


def monte_carlo_l_step_rollout_policy(base_pi: Union[List, MaskablePPO], s: int, gamma: float,
                                      batch_size: int, states: List, actions: List, c_max: float,
                                      p_max: float, N: List, E: List, S: List, V_base: List, l: int,
                                      delay_table: List, deterministic_model: bool = False, env=None) \
        -> Tuple[List, List, int]:
    """
    Computes the monte-Carlo l-step rollout policy

    :param base_pi: the base policy
    :param s: the current state
    :param gamma: the discount factor
    :param batch_size: the batch size for evaluation
    :param states: the state space
    :param actions: the action space
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :param N: the set of nodes
    :param E: the set of edges
    :param S: the set of services
    :param V_base: the value function of the base polichy
    :param l: the lookahead optimization horizon
    :param delay_table: the table with the predicted delays
    :param deterministic_model: boolean flag indicating whether it is a deterministic model or not
    :param env: the environment for evaluation
    :return: the rollout policy and the value function of the rollout policy and the best action in the current state
    """
    if deterministic_model:
        a, rollout_val = deterministic_l_step_lookahead_maximization(
            s=s, gamma=gamma, states=states, actions=actions, c_max=c_max, p_max=p_max, N=N, E=E, S=S, V_base=V_base,
            l=l, delay_table=delay_table)
    else:
        a, rollout_val = monte_carlo_l_step_lookahead_maximization(s=s, gamma=gamma, batch_size=batch_size,
                                                                   states=states,
                                                                   actions=actions, c_max=c_max, p_max=p_max, N=N, E=E,
                                                                   S=S,
                                                                   V_base=V_base, l=l, delay_table=delay_table)
    if isinstance(base_pi, list) or isinstance(base_pi, np.ndarray):
        rollout_pi = base_pi.copy()
        rollout_pi[s] = np.zeros(len(base_pi[s]))
        rollout_pi[s][a] = 1
    else:
        rollout_pi = base_pi
    V_rollout = V_base.copy()
    V_rollout[s] = rollout_val
    return rollout_pi, V_rollout, a


def monte_carlo_l_step_multiagent_rollout_policy(base_pi: Union[List, MaskablePPO], s: int, gamma: float,
                                                 batch_size: int, states: List, actions: List, c_max: float,
                                                 p_max: float, N: List, E: List, S: List, V_base: List, l: int,
                                                 component_spaces,
                                                 delay_table: List, deterministic_model: bool = False, env=None) \
        -> Tuple[List, List, int]:
    """
    Computes the Monte-Carlo l-step multi-agent rollout policy

    :param base_pi: the base policy
    :param s: the current state
    :param gamma: the discount factor
    :param batch_size: the batch size
    :param states: the state space
    :param actions: the action space
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :param N: the set of nodes
    :param E: the set of edges
    :param S: the set of services
    :param V_base: the value function of the base policy
    :param l: the lookahead horizon
    :param component_spaces: action spaces of differet action components
    :param delay_table: table with predicted service request delays
    :param deterministic_model: boolean flag indicating whether it is a deterministic model or not
    :param env: the environment for evaluation
    :return: the rollout policy and the value function of the rollout policy and the best action in the current state
    """
    if deterministic_model:
        partial_a = []
        rollout_val = 0
        for agent_index in range(len(component_spaces)):
            a, rollout_val = deterministic_l_step_lookahead_maximization_multiagent(
                s=s, gamma=gamma, states=states, actions=actions, c_max=c_max, p_max=p_max, N=N, E=E, S=S,
                V_base=V_base,
                l=l, delay_table=delay_table, component_spaces=component_spaces, base_pi=base_pi,
                agent_index=agent_index, partial_a=partial_a)
            partial_a.append(a[agent_index])
        a = actions.index(tuple(partial_a))
    else:
        a, rollout_val = monte_carlo_l_step_lookahead_maximization(s=s, gamma=gamma, batch_size=batch_size,
                                                                   states=states,
                                                                   actions=actions, c_max=c_max, p_max=p_max, N=N, E=E,
                                                                   S=S,
                                                                   V_base=V_base, l=l, delay_table=delay_table)
    if isinstance(base_pi, list) or isinstance(base_pi, np.ndarray):
        rollout_pi = base_pi.copy()
        rollout_pi[s] = np.zeros(len(base_pi[s]))
        rollout_pi[s][a] = 1
    else:
        rollout_pi = base_pi
    V_rollout = V_base.copy()
    V_rollout[s] = rollout_val
    return rollout_pi, V_rollout, a


def deterministic_l_step_lookahead_maximization(s: int, gamma: float, states: List, actions: List, c_max: float,
                                                p_max: float, N: List, E: List, S: List, V_base: List, l: int,
                                                delay_table: List) -> Tuple[int, float]:
    """
    Performs the deterministic l-step lookahead maximization

    :param s: the current state
    :param gamma: the discount factor
    :param states: the state space
    :param actions: the action space
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :param N: the set of nodes
    :param E: the set of edges
    :param S: the set of services
    :param V_base: the value function of the base policy
    :param l: the length of the lookahead horizon
    :param delay_table: a table with the predicted delays
    :return: the best action and its Q-factor
    """
    if l == 1:
        A = np.zeros(len(actions))
        for a in range(len(actions)):
            r = reward_function(s=states[int(s)], a=actions[int(a)], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                services=S, states=states, delay_table=delay_table)
            s_prime = transition_function(s=states[int(s)], a=actions[a], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                          services=S)
            s_prime = states.index(s_prime)
            A[a] = r + gamma * V_base[s_prime]
        best_a = np.argmax(A)
        return int(best_a), float(A[best_a])
    else:
        A = np.zeros(len(actions))
        for a in range(len(actions)):
            r = reward_function(s=states[int(s)], a=actions[int(a)], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                services=S, states=states, delay_table=delay_table)
            s_prime = transition_function(s=states[int(s)], a=actions[a], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                          services=S)
            s_prime = states.index(s_prime)
            _, s_prime_val = deterministic_l_step_lookahead_maximization(
                s=s_prime, gamma=gamma, states=states, actions=actions, c_max=c_max, p_max=p_max, N=N,
                E=E, S=S, V_base=V_base, l=l - 1, delay_table=delay_table)
            A[a] = r + gamma * s_prime_val
        best_a = np.argmax(A)
        return int(best_a), float(A[best_a])


def deterministic_l_step_lookahead_maximization_multiagent(s: int, gamma: float, states: List, actions: List,
                                                           c_max: float,
                                                           p_max: float, N: List, E: List, S: List, V_base: List,
                                                           l: int,
                                                           delay_table: List, component_spaces, agent_index: int,
                                                           base_pi, partial_a) -> Tuple[int, float]:
    """
    Performs the deterministic l-step multi-agent lookahead maximization

    :param s: the current state
    :param gamma: the discount factor
    :param states: the state space
    :param actions: the action space
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :param N: the set of nodes
    :param E: the set of edges
    :param S: the set of services
    :param V_base: the value function of the base policy
    :param l: the length of the lookahead horizon
    :param delay_table: a table with the predicted delays
    :param component_spaces: action spaces for each component of the action space
    :param agent_index: the index of the agent
    :param base_pi: the base policy
    :param partial_a: the previous comleted action components by other agents
    :return: the best action and its Q-factor
    """
    if l == 1:
        A = np.zeros(len(component_spaces[agent_index]))
        base_action_idx = np.random.choice(np.arange(0, len(actions)), p=base_pi[s])
        base_action = actions[base_action_idx]
        a_vec = partial_a.copy()
        a_vec.append(-1)
        for i in range(agent_index + 1, len(component_spaces)):
            a_vec.append(base_action[i])
        for local_a in range(len(component_spaces[agent_index])):
            a_vec[agent_index] = component_spaces[agent_index][local_a]
            a = actions.index(tuple(a_vec))
            r = reward_function(s=states[int(s)], a=actions[int(a)], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                services=S, states=states, delay_table=delay_table)
            s_prime = transition_function(s=states[int(s)], a=actions[a], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                          services=S)
            s_prime = states.index(s_prime)
            A[local_a] = r + gamma * V_base[s_prime]
        best_local_a_idx = np.argmax(A)
        best_a = a_vec.copy()
        best_a[agent_index] = component_spaces[agent_index][best_local_a_idx]
        return best_a, A[best_local_a_idx]
    else:
        A = np.zeros(len(component_spaces[agent_index]))
        base_action_idx = np.random.choice(np.arange(0, len(actions)), p=base_pi[s])
        base_action = actions[base_action_idx]
        a_vec = partial_a.copy()
        a_vec.append(-1)
        for i in range(agent_index + 1, len(component_spaces)):
            a_vec.append(base_action[i])
        for local_a in range(len(component_spaces[agent_index])):
            a_vec[agent_index] = component_spaces[agent_index][local_a]
            a = actions.index(tuple(a_vec))
            r = reward_function(s=states[int(s)], a=actions[int(a)], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                services=S, states=states, delay_table=delay_table)
            s_prime = transition_function(s=states[int(s)], a=actions[a], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                          services=S)
            s_prime = states.index(s_prime)
            _, s_prime_val = deterministic_l_step_lookahead_maximization_multiagent(
                s=s_prime, gamma=gamma, states=states, actions=actions, c_max=c_max, p_max=p_max, N=N,
                E=E, S=S, V_base=V_base, l=l - 1, delay_table=delay_table, component_spaces=component_spaces,
                partial_a=partial_a, agent_index=agent_index, base_pi=base_pi)
            A[local_a] = r + gamma * s_prime_val
        best_local_a_idx = np.argmax(A)
        best_a = a_vec.copy()
        best_a[agent_index] = component_spaces[agent_index][best_local_a_idx]
        return best_a, A[best_local_a_idx]


def monte_carlo_l_step_lookahead_maximization(s: int, gamma: float, batch_size: int, states: List,
                                              actions: List, c_max: float, p_max: float, N: List, E: List, S: List,
                                              V_base: List, l: int, delay_table: List) -> Tuple[int, float]:
    """
    Monte-Carlo l-step lookahead maximization

    :param s: the current state
    :param gamma: the discount factor
    :param batch_size: the batch size for evaluation
    :param states: the state space
    :param actions: the action space
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :param N: the set of nodes
    :param E: the set of edges
    :param S: the set of services
    :param V_base: the value function of the base policy
    :param l: the length of the lookahead horizon
    :param delay_table: the table with the predicted delays
    :return: the best action and its value
    """
    V_base_prime = V_base.copy()
    for i in range(0, l - 1):
        for s_k in range(len(states)):
            print(f" l-step lookahead, time-index:{i}/{l}, state: {s_k}/{len(states)}")
            A = np.zeros(len(actions))
            for a in range(len(actions)):
                returns = []
                for k in range(batch_size):
                    r = reward_function(s=states[int(s_k)], a=actions[int(a)], nodes=N, edges=E, c_max=c_max,
                                        p_max=p_max, services=S, delay_table=delay_table, states=states)
                    s_prime = transition_function(s=states[int(s_k)], a=actions[a], nodes=N, edges=E, c_max=c_max,
                                                  p_max=p_max, services=S)
                    s_prime = states.index(s_prime)
                    returns.append(r + gamma * V_base_prime[s_prime])
                A[a] = np.mean(returns)
            V_base_prime[s_k] = np.max(A)

    A = np.zeros(len(actions))
    for a in range(len(actions)):
        returns = []
        for k in range(batch_size):
            r = reward_function(s=states[int(s)], a=actions[int(a)], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                services=S, delay_table=delay_table, states=states)
            s_prime = transition_function(s=states[int(s)], a=actions[a], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                          services=S)
            s_prime = states.index(s_prime)
            returns.append(r + gamma * V_base_prime[s_prime])
        A[a] = np.mean(returns)
    best_a = np.argmax(A)
    return int(best_a), float(A[best_a])


def monte_carlo_policy_evaluation(policy: Union[MaskablePPO, List], states: List, actions: List, gamma: float,
                                  batch_size: int, N: List,
                                  E: List, c_max: float, p_max: float, S: List, delay_table: List,
                                  delta_c: float, delta_p: float, eval_state=None,
                                  modified_s: int = None, modified_a: int = None, env=None) -> List:
    """
    Performs Monte-Carlo policy evaluation

    :param policy: the policy to evaluate
    :param states: the state space
    :param actions: the action space
    :param gamma: the discount factor
    :param batch_size: the batch size for evaluation
    :param N: the set of nodes
    :param E: the set of edges
    :param c_max: the maximum value of the scaling action
    :param p_max: the maximum value of the routing action
    :param S: the set of services
    :param delay_table: the table with the predicted service request delays
    :param delta_c: the discretization step of the scaling action
    :param delta_p: the discretization step of the routing action
    :param eval_state: the state to use for evaluation
    :param modified_s: a possibly modified state to evaluate
    :param modified_a: a possibly modified action to evaluate
    :param env: the evaluation RL environment
    :return: the value function
    """
    V = np.zeros(len(states))
    horizon = int(1 / (1 - gamma))
    if eval_state is None:
        eval_states = list(range(len(states)))
    else:
        eval_states = [eval_state]
    for s in eval_states:
        returns = []
        for i in range(batch_size):
            ret = 0
            state = s
            for t in range(horizon):
                if modified_s is not None and modified_s == state:
                    a = modified_a
                else:
                    if isinstance(policy, list) or isinstance(policy, np.ndarray):
                        a = np.random.choice(np.arange(0, len(actions)), p=policy[state])
                    else:
                        obs = convert_dp_s_to_env_s(s=states[state])
                        env.set_state(p1=obs[0], p2=obs[1], cpu1=obs[2], cpu2=obs[3])
                        action_masks = get_action_masks(env)
                        a, _ = policy.predict(obs, deterministic=True, action_masks=action_masks)
                        a = convert_env_a_to_dp_a(a=list(a), delta_p=delta_p, delta_c=delta_c)
                        a = actions.index(tuple(a))
                r = reward_function(s=states[int(state)], a=actions[int(a)], nodes=N, edges=E, c_max=c_max, p_max=p_max,
                                    services=S, delay_table=delay_table, states=states)
                # print(f"s: {states[int(state)]}, a: {actions[int(a)]}, r: {r}")
                ret += r
                s_prime = transition_function(s=states[int(state)], a=actions[int(a)], nodes=N, edges=E, c_max=c_max,
                                              p_max=p_max, services=S)
                state = states.index(s_prime)
            returns.append(ret)
        V[s] = np.mean(returns)
    return list(V.tolist())
