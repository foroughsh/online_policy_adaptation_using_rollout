from gym.envs.registration import register

register(
    id='routing-env-v1',
    entry_point='scenario_1_env.routing_env:RoutingEnv'
)