import numpy as np
import gymnasium as gym
from typing import Dict, SupportsFloat

from feature4irl.util.feature_gen import select_feat_extractor
from gymnasium.wrappers.flatten_observation import FlattenObservation


##################################################
##      TransformRewardLearnedCont
###################################################
class TransformRewardLearnedCont(gym.RewardWrapper):
    """Transform the reward via an arbitrary function.
    # FIXME reward_range if this warning is important??
    Warning:
        If the base environment specifies a reward range which is not invariant 
        under :attr:`f`, the :attr:`reward_range` of the wrapped environment will be incorrect.
    """

    def __init__(self, env: gym.Env, alpha=None, configs=None):
        """Initialize the :class:`TransformReward` wrapper with an environment and reward transform function :attr:`f`.
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
##      PendulumWrapper
###################################################
class PendulumWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for pendulum env
    """
    def __init__(
        self,
        env: gym.Env,
        reward_path = None,
        env_name = None,
        configs = None,
        scaler_path = None,

        ) -> None:
        
        env = StoreObservation(env)
        
        if reward_path == 'None':
            pass
        elif reward_path == None:
            raise Exception('reward path cannnot be None') 
        else:
            alpha = np.load(reward_path + '.npy')
            scaler_params = np.load(scaler_path + '.npy', allow_pickle=True)

            if configs == None:
                raise Exception('configs cannnot be None') 
            else: 
                configs['scaler_params'] = scaler_params.tolist()
                env = TransformRewardLearnedCont(env, alpha, configs)

        super().__init__(env)
        
        
        
##################################################
##      CartPoleWrapper
###################################################
class CartPoleWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for pendulum env
    """
    def __init__(
        self,
        env: gym.Env,
        reward_path = None,
        env_name = None,
        configs = None,
        scaler_path = None,
        
        ) -> None:
                
        env = StoreObservation(env)
        
        if reward_path == 'None':
            pass
        elif reward_path == None:
            raise Exception('reward path cannnot be None') 
        else:
            alpha = np.load(reward_path + '.npy')
            scaler_params = np.load(scaler_path + '.npy', allow_pickle=True)
            if configs == None:
                raise Exception('configs cannnot be None') 
            else: 
                configs['scaler_params'] = scaler_params.tolist()
                env = TransformRewardLearnedCont(env, alpha, configs)  

        super().__init__(env)



##################################################
##      ACROBOTWrapper
###################################################
class AcrobotWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for pendulum env
    """
    def __init__(
        self,
        env: gym.Env,
        reward_path = None,
        env_name = None,
        configs = None,
        scaler_path = None,
        
        ) -> None:
        
        env = StoreObservation(env)
        
        if reward_path == 'None':
            pass
        elif reward_path == None:
            raise Exception('reward path cannnot be None') 
        else:
            alpha = np.load(reward_path + '.npy')
            scaler_params = np.load(scaler_path + '.npy', allow_pickle=True)

            if configs == None:
                raise Exception('configs cannnot be None') 
            else: 
                configs['scaler_params'] = scaler_params.tolist()
                env = TransformRewardLearnedCont(env, alpha, configs)

        super().__init__(env)
        


##################################################
##      HighwayWrapper
###################################################

class HighwayWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for pendulum env
    """
    def __init__(
        self,
        env: gym.Env,
        reward_path = None,
        env_name = None,
        d_states_env = None, 
        env_config = None,
        configs = None,

        ) -> None:

        env.configure(env_config["config"])
        env.reset()

        # env.configure({"offscreen_rendering": True})
        env.observation_space = gym.spaces.Box(low=float("-inf"), 
                                               high=float("inf"), 
                                               shape=(d_states_env, ), 
                                               dtype=np.float32)

        
        env.reset()
        env = FlattenObservation(env)
        env = StoreObservation(env)
        
        # TODO check if seeding needed
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        
        if reward_path == 'None':
            pass
        elif reward_path == None:
            raise Exception('reward path cannnot be None') 
        else:
            alpha = np.load(reward_path + '.npy')
            if configs == None:
                raise Exception('configs cannnot be None') 
            else: 
                env = TransformRewardLearnedCont(env, alpha, configs)
        
        super().__init__(env)




##################################################
##      HopperWrapper
###################################################
class HopperWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for Hopper env
    """
    def __init__(
        self,
        env: gym.Env,
        reward_path = None,
        env_name = None,
        configs = None,

        ) -> None:
        
        env = StoreObservation(env)
        
        if reward_path == 'None':
            pass
        elif reward_path == None:
            raise Exception('reward path cannnot be None') 
        else:
            alpha = np.load(reward_path + '.npy')
            if configs == None:
                raise Exception('configs cannnot be None') 
            else: 
                env = TransformRewardLearnedCont(env, alpha, configs)

        super().__init__(env)



##################################################
##      
###################################################
class LunarLanderWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for pendulum env
    """
    def __init__(
        self,
        env: gym.Env,
        reward_path = None,
        env_name = None,
        configs = None,
        scaler_path = None,
        
        ) -> None:
       
        # import pdb; pdb.set_trace()
         
        env = StoreObservation(env)
        
        if reward_path == 'None':
            pass
        elif reward_path == None:
            raise Exception('reward path cannnot be None') 
        else:
            alpha = np.load(reward_path + '.npy')
            scaler_params = np.load(scaler_path + '.npy', allow_pickle=True)

            if configs == None:
                raise Exception('configs cannnot be None') 
            else: 
                
                configs['scaler_params'] = scaler_params.tolist()
                env = TransformRewardLearnedCont(env, alpha, configs)

        super().__init__(env)
        
        
##################################################
##      HopperWrapper
###################################################
class HopperWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for Hopper env
    """
    def __init__(
        self,
        env: gym.Env,
        reward_path = None,
        env_name = None,
        configs = None,
        scaler_path = None,

        ) -> None:
        
        env = StoreObservation(env)
        
        if reward_path == 'None':
            pass
        elif reward_path == None:
            raise Exception('reward path cannnot be None') 
        else:
            alpha = np.load(reward_path + '.npy')
            scaler_params = np.load(scaler_path + '.npy', allow_pickle=True)

            if configs == None:
                raise Exception('configs cannnot be None') 
            else: 

                configs['scaler_params'] = scaler_params.tolist()
                env = TransformRewardLearnedCont(env, alpha, configs)

        super().__init__(env)
        
        
##################################################
##      ReacherWrapper
###################################################
class ReacherWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Wrapper for pendulum env
    """
    def __init__(
        self,
        env: gym.Env,
        reward_path = None,
        env_name = None,
        configs = None,
        scaler_path = None,
        ) -> None:
        
        env = StoreObservation(env)
        
        if reward_path == 'None':
            pass
        elif reward_path == None:
            raise Exception('reward path cannnot be None') 
        else:
            alpha = np.load(reward_path + '.npy')
            scaler_params = np.load(scaler_path + '.npy', allow_pickle=True)
            
            if configs == None:
                raise Exception('configs cannnot be None') 
            else:
                configs['scaler_params'] = scaler_params.tolist()
                env = TransformRewardLearnedCont(env, alpha, configs)

        super().__init__(env)
