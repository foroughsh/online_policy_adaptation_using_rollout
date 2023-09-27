from gym.envs.registration import register

register(
    id='routing-env-v3',
    entry_point='scenario_3_env.routing_env:RoutingEnv'
)