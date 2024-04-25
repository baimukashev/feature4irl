import numpy as np
import gymnasium as gym

from feature4irl.util.feature_gen import select_feat_extractor


##################################################
#      TransformRewardLearnedCont
###################################################
class TransformRewardLearnedCont(gym.RewardWrapper):
    """Transform the reward via an arbitrary function."""
    def __init__(self, env: gym.Env, alpha=None, configs=None):
        """Initialize the :class:`TransformReward` wrapper with an environment
        and reward transform function :attr:`f`.
        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the reward
        """
        super().__init__(env)

        self.alpha = alpha
        self.configs = configs

    def reward(self, reward):
        """Transforms the reward using callable :attr:`f`.
        Args:
            reward: The reward to transform
        Returns:
            The transformed reward
        """
        state = self.temp_state
        env_name = self.spec.id
        feature_expectations = select_feat_extractor(env_name,
                                                     state,
                                                     cfg=self.configs)
        reward = feature_expectations.dot(self.alpha)
        return reward


class StoreObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        """Resizes image observations to shape given by :attr:`shape`.
        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        super().__init__(env)

    def observation(self, observation):
        self.temp_state = observation
        return observation


class StoreAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        """Resizes image observations to shape given by :attr:`shape`.
        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        super().__init__(env)

    def action(self, action):
        self.temp_action = action
        return action


class BaseEnvWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Base wrapper to handle observation storage and conditional reward transformation.
    """
    def __init__(self, env: gym.Env, reward_path=None, scaler_path=None, configs=None):
        # Initialize base environment
        super().__init__(StoreObservation(env))
        
        # Handle reward transformation if paths and configs are provided
        if reward_path and reward_path != 'None' and scaler_path and configs:
            alpha = np.load(reward_path + '.npy')
            scaler_params = np.load(scaler_path + '.npy', allow_pickle=True)
            configs['scaler_params'] = scaler_params.tolist()
            self.env = TransformRewardLearnedCont(self.env, alpha, configs)
        elif reward_path is None:
            raise ValueError('reward path cannot be None if specified')

class PendulumWrapper(BaseEnvWrapper):
    """
    Specific wrapper for the Pendulum environment.
    """
    def __init__(self, env, reward_path=None, env_name=None, scaler_path=None, configs=None):
        super().__init__(env, reward_path, scaler_path, configs)

class CartPoleWrapper(BaseEnvWrapper):
    """
    Specific wrapper for the CartPole environment.
    """
    def __init__(self, env, reward_path=None, env_name=None, scaler_path=None, configs=None):
        super().__init__(env, reward_path, scaler_path, configs)
        
class AcrobotWrapper(BaseEnvWrapper):
    """
    Specific wrapper for the CartPole environment.
    """
    def __init__(self, env, reward_path=None, env_name=None, scaler_path=None, configs=None):
        super().__init__(env, reward_path, scaler_path, configs)