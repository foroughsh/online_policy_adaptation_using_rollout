import numpy as np
from sb3_contrib import MaskablePPO
from online_policy_adaptation_using_rollout.system_model.system_model_scenario_1 import (
    state_space, action_space, services, edges, nodes)
from online_policy_adaptation_using_rollout.ppo_base_policy.scenario_1_env.routing_env import RoutingEnv
from online_policy_adaptation_using_rollout.evaluation.evaluate_scenario_1 import (
    compute_value_function_of_policy, compute_delay_table, random_base_policy, greedy_base_policy,
    optimal_base_policy, run_vi, run_pi, run_exact_one_step_online_play, run_monte_carlo_one_step_online_play,
    run_base_policy_online, newton_step_data)

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
