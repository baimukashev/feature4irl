from abc import ABCMeta
from abc import abstractmethod

from stable_baselines3.common.env_util import make_vec_env
from feature4irl.envs.wrappers import *


class BaseAlgo(object, metaclass=ABCMeta):
    """Abstract agent class."""

    def __init__(self, cfg) -> None:

        self.cfg = cfg
        self.cfg["wrapper_kwargs"]["configs"] = cfg

    def create_env(self):
        envs = make_vec_env(
            self.cfg["env_name"],
            n_envs=self.cfg["n_envs"],
            wrapper_class=eval(self.cfg["wrapper_class"]),
            wrapper_kwargs=self.cfg["wrapper_kwargs"],
            # vec_env_cls=SubprocVecEnv
        )
        return envs

    def create_agent(self, use_init_params):
        if self.cfg["agent_name"] == "sb_ppo":
            from feature4irl.agents.sb3_ppo import PPOAgent

            agent = PPOAgent(self.cfg, env=self.env, use_init_params=use_init_params)
        elif self.cfg["agent_name"] == "sb_sac":
            from feature4irl.agents.sb3_sac import SACAgent

            agent = SACAgent(self.cfg, env=self.env, use_init_params=use_init_params)
        else:
            raise NotImplementedError

        return agent

    @abstractmethod
    def log_data(self, data):
        """Select an action for training.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        """Select an action for training.

        Returns:
            ~object: action
        """
        raise NotImplementedError()
