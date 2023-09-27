from gym.envs.registration import register

register(
    id='routing-env-v1',
    entry_point='online_policy_adaptation_using_rollout.ppo_base_policy.scenario_1_env.routing_env:RoutingEnv',
    kwargs={'path_to_system_model': None}
)