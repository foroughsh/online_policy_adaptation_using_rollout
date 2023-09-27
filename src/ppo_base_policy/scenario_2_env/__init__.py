from gym.envs.registration import register

register(
    id='routing-env-v2',
    entry_point='scenario_2_env.routing_env:RoutingEnv'
)