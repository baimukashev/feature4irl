import torch
from stable_baselines3.sac import SAC
from feature4irl.agents.base_agent import Agent


class SACAgent(Agent):

    def __init__(self, cfg, env, use_init_params) -> None:
        super().__init__(cfg)
        self.configs = cfg
        self.env = env

        if use_init_params:
            pi_size = self.cfg["init_pi_size"]
            vf_size = self.cfg["init_pi_size"]
            gamma = self.cfg["init_gamma"]
            learning_rate = self.cfg["init_learning_rate"]
            self.total_timesteps = self.cfg["init_total_timesteps"]
        else:
            pi_size = self.cfg["pi_size"]
            vf_size = self.cfg["pi_size"]
            gamma = self.cfg["gamma"]
            learning_rate = self.cfg["learning_rate"]
            self.total_timesteps = self.cfg["total_timesteps"]

        self.policy = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                net_arch=dict(qf=[pi_size, pi_size], pi=[vf_size, vf_size])
            ),
            # batch_size=self.cfg["batch_size"],
            # learning_rate=learning_rate,
            gamma=gamma,
            # learning_starts=10000,
            # buffer_size=100000,
            # tau=0.01,
            # ent_coef=0.2,
            # train_freq=2,
            # use_sde=self.cfg["use_sde"],
            # sde_sample_freq=self.cfg["sde_sample_freq"],
            verbose=self.cfg["verbose"],
            tensorboard_log=f"checkpoints/{self.cfg['exp_name']}/logs/",
            device=torch.device(
                self.cfg["device"],
            ),
        )

        # set additional parameters if needed
        if self.configs.get("max_grad_norm"):
            self.policy.max_grad_norm = self.configs["max_grad_norm"]

        if self.configs.get("vf_coef"):
            self.policy.vf_coef = self.configs["vf_coef"]

        if self.configs.get("ent_coef"):
            self.policy.ent_coef = self.configs["ent_coef"]

        if self.configs.get("clip_range"):
            self.policy.clip_range = self.configs["clip_range"]

    def act(self, obs):
        action, *_ = self.policy.predict(torch.tensor(obs).float())

        return action

    def learn(self, logname=None):
        self.policy.learn(total_timesteps=self.total_timesteps, tb_log_name=logname)

    def save(self, path):
        self.policy.save(path)

    def load(self, path, env=None, custom_objects=None):
        self.policy = SAC.load(path, env=env, custom_objects=custom_objects)

        return self.policy
