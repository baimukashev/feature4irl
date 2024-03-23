from gymnasium.envs.registration import register

register(
    id="Pendulum-m1",
    entry_point='gymnasium.envs.classic_control.pendulum:PendulumEnv', 
    reward_threshold=None, 
    nondeterministic=False, 
    max_episode_steps=200, 
    order_enforce=True, 
    autoreset=False, 
    disable_env_checker=False, 
    apply_api_compatibility=False, 
    kwargs={}, 
    # additional_wrappers=(), 
    # vector_entry_point=None
)

register(
    id="CartPole-m1",
    entry_point='gymnasium.envs.classic_control.cartpole:CartPoleEnv', 
    reward_threshold=None, 
    nondeterministic=False, 
    max_episode_steps=200, 
    order_enforce=True, 
    autoreset=False, 
    disable_env_checker=False, 
    apply_api_compatibility=False, 
    kwargs={}, 
    # additional_wrappers=(), 
    # vector_entry_point=None
)

# import highway_env
# highway_env.register_highway_envs()