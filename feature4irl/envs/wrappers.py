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

        feature_expectations = select_feat_extractor(env_name, state, cfg=self.configs)

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


##################################################
#      PendulumWrapper
###################################################
class PendulumWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for pendulum env
    """

    def __init__(
        self,
        env: gym.Env,
        reward_path=None,
        env_name=None,
        configs=None,
        scaler_path=None,
    ) -> None:

        env = StoreObservation(env)

        if reward_path == "None":
            pass
        elif reward_path is None:
            raise Exception("reward path cannnot be None")
        else:
            alpha = np.load(reward_path + ".npy")
            scaler_params = np.load(scaler_path + ".npy", allow_pickle=True)

            if configs is None:
                raise Exception("configs cannnot be None")
            else:
                configs["scaler_params"] = scaler_params.tolist()
                env = TransformRewardLearnedCont(env, alpha, configs)

        super().__init__(env)


##################################################
#      CartPoleWrapper
###################################################
class CartPoleWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for pendulum env
    """

    def __init__(
        self,
        env: gym.Env,
        reward_path=None,
        env_name=None,
        configs=None,
        scaler_path=None,
    ) -> None:

        env = StoreObservation(env)

        if reward_path == "None":
            pass
        elif reward_path is None:
            raise Exception("reward path cannnot be None")
        else:
            alpha = np.load(reward_path + ".npy")
            scaler_params = np.load(scaler_path + ".npy", allow_pickle=True)
            if configs is None:
                raise Exception("configs cannnot be None")
            else:
                configs["scaler_params"] = scaler_params.tolist()
                env = TransformRewardLearnedCont(env, alpha, configs)

        super().__init__(env)


##################################################
#      ACROBOTWrapper
###################################################
class AcrobotWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for pendulum env
    """

    def __init__(
        self,
        env: gym.Env,
        reward_path=None,
        env_name=None,
        configs=None,
        scaler_path=None,
    ) -> None:

        env = StoreObservation(env)

        if reward_path == "None":
            pass
        elif reward_path is None:
            raise Exception("reward path cannnot be None")
        else:
            alpha = np.load(reward_path + ".npy")
            scaler_params = np.load(scaler_path + ".npy", allow_pickle=True)

            if configs is None:
                raise Exception("configs cannnot be None")
            else:
                configs["scaler_params"] = scaler_params.tolist()
                env = TransformRewardLearnedCont(env, alpha, configs)

        super().__init__(env)
