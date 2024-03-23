from abc import ABCMeta
from abc import abstractmethod

import gymnasium as gym

# class BaseEnv(gym.Env, metaclass=ABCMeta):
#     """Abstract agent class."""

#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = cfg
    
#     # @abstractmethod
#     def make_env(self):
#         env = gym.make(self.cfg["env_name"])
    
#         return env   

#     # @abstractmethod
#     def make_env_vec(self, n_envs=4, source='stable'):
#         if source == 'stable':
#             from stable_baselines3.common.env_util import make_vec_env
#             vec_env = make_vec_env(self.cfg["env_name"], n_envs=4)
#         else: 
#             raise NotImplementedError
            
#         return vec_env
    
    
    
#         pass
