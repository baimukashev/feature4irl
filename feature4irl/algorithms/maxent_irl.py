import os
import numpy as np
import random
import csv
import ot
import wandb
from sklearn.preprocessing import RobustScaler
from hydra.core.hydra_config import HydraConfig
from joblib import Parallel, delayed

from feature4irl.algorithms.base_algo import BaseAlgo
from feature4irl.util.feature_gen import (
    find_feature_expectations,
    generate_trajectories,
)


class ContMaxEntIRL(BaseAlgo):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def train(self):
        self.configure_experiment()

        self.env = self.create_env()
        self.rollout_env = self.create_env()

        self.expert = self.create_agent(use_init_params=True)  # CHECK
        self.expert.policy.tensorboard_log = f"checkpoints/{self.exp_name}/logs/"

        # Train or load the expert POLICY
        #####################################################################
        if self.cfg["load_expert"]:
            self.expert.load(
                self.cfg["path_to_expert"],
                env=self.env,
                custom_objects={
                    "observation_space": self.env.observation_space,
                    "action_space": self.env.action_space,
                },
            )

            with open(f"checkpoints/{self.exp_name}/readme.txt", "w") as f:
                f.write("This run uses the saved model from " +
                        f"{self.cfg['path_to_expert']}")

        else:
            self.expert.learn(logname="Expert")
            self.expert.save(f"checkpoints/{self.exp_name}/files/ppo_expert")

        if self.cfg["testing"]:
            video_path = f"checkpoints/{self.exp_name}/files/ppo_expert"
            expert_results = self.expert.save_render(
                video_dir=video_path,
                test_num=self.cfg["test_num"],
                test_env=self.rollout_env,
            )

        # Generate or load the DATA
        #####################################################################
        if self.cfg["load_data"]:
            expert_trajs = np.load(
                f'{self.cfg["path_to_data"]}expert_trajs_{self.cfg["n_trajs"]}.npy'
            )
            expert_ts = np.load(
                f'{self.cfg["path_to_data"]}expert_ts_{self.cfg["n_trajs"]}.npy'
            )
            with open(f"checkpoints/{self.exp_name}/readme.txt", "w") as f:
                f.write(
                    f'This run uses the saved data of N={self.cfg["n_trajs"]} trajs from {self.cfg["path_to_data"]}'
                )

        else:
            # Collect data
            expert_trajs, expert_ts, rs = self.collect_rollouts(
                self.expert.policy)

            np.save(
                f'checkpoints/{self.exp_name}/files/expert_trajs_{self.cfg["n_trajs"]}',
                expert_trajs,
            )
            np.save(
                f'checkpoints/{self.exp_name}/files/expert_ts_{self.cfg["n_trajs"]}',
                expert_ts,
            )
            np.save(
                f'checkpoints/{self.exp_name}/files/expert_rs_{self.cfg["n_trajs"]}',
                rs)

        print("Expert  data  ", expert_trajs.shape, expert_ts.shape)
        # print("Expert mean  ", np.mean(expert_trajs, axis=0).tolist())
        # print("Expert  std  ", np.std(expert_trajs, axis=0).tolist())

        robust_scaler = RobustScaler()
        robust_scaler.fit(expert_trajs)
        scaler_parameters = {
            "center": robust_scaler.center_.tolist(),
            "scale": robust_scaler.scale_.tolist(),
        }
        self.cfg["scaler_params"] = scaler_parameters

        scaler_path = f"checkpoints/{self.exp_name}/files/scaler_temp"
        np.save(scaler_path, scaler_parameters)
        self.cfg["wrapper_kwargs"]["scaler_path"] = scaler_path

        if self.cfg["bc_only"]:
            return 0

        ######################################################################
        # IRL part
        #####################################################################
        # IRL initialize
        lr = self.cfg["lr"]
        epochs = self.cfg["epochs"]
        d_states = self.cfg["d_states"]

        # init weights
        self.alpha = np.array([random.uniform(-1, 1) for x in range(d_states)])
        alpha_path = f"checkpoints/{self.exp_name}/files/alpha_temp"
        np.save(alpha_path, self.alpha)

        print("\n....  IRL training ")
        feature_expectations_expert = find_feature_expectations(
            self.cfg, expert_trajs, expert_ts)

        # train
        for epoch in range(1, epochs + 1):

            # updata env and agents
            self.cfg["wrapper_kwargs"]["reward_path"] = alpha_path
            self.cfg["wrapper_kwargs"]["scaler_path"] = scaler_path

            self.env = self.create_env()
            # !! use random agent every time
            self.agent = self.create_agent(use_init_params=True)  # CHECK
            self.agent.learn(logname="Agent")
            agent_trajs, agent_ts, _ = self.collect_rollouts(self.agent.policy)

            # # theta update
            feature_expectations_learner = find_feature_expectations(
                self.cfg, agent_trajs, agent_ts)
            grad = feature_expectations_expert - feature_expectations_learner

            # for sanity check
            ratio = self.cfg["ratio_divergence"]
            expert_size = int(expert_trajs.shape[0] * ratio)
            agent_size = int(agent_trajs.shape[0] * ratio)
            dist = 0

            # # # log
            if epoch % self.cfg["log_freq"] == 0 and self.cfg["track"]:
                self.log_data(epoch, self.alpha, grad, dist, lr)

            # update
            self.alpha += lr * grad
            lr *= self.cfg["alpha_decay"]

            # save alpha for env
            np.save(alpha_path, self.alpha)
            if epoch % self.cfg["save_freq"] == 0 or epoch == 1:
                np.save(
                    f"checkpoints/{self.exp_name}/files/alpha_ep" + str(epoch),
                    self.alpha,
                )
                self.agent.save(
                    f"checkpoints/{self.exp_name}/files/ppo_learned_reward_ep{epoch}"
                )
                np.save(
                    f"checkpoints/{self.exp_name}/files/agent_trajs_learned" +
                    str(epoch),
                    agent_trajs,
                )
                np.save(
                    f"checkpoints/{self.exp_name}/files/agent_ts_learned" +
                    str(epoch),
                    agent_ts,
                )

            # run testing and save videos
            if self.cfg["testing"] and (epoch % self.cfg["test_epoch"] == 0
                                        or epoch == 1):
                video_path = (
                    f"checkpoints/{self.exp_name}/files/ppo_learned_reward_ep{epoch}"
                )
                agent_results = self.agent.save_render(
                    video_dir=video_path,
                    test_num=self.cfg["test_num"],
                    test_env=self.rollout_env,
                )

                self.run.log({"avg_test_reward/MEAN_agent": agent_results[0]},
                             commit=False)
                self.run.log({"avg_test_reward/STD_agent": agent_results[1]},
                             commit=False)
                self.run.log(
                    {"avg_test_reward/MEAN_expert": expert_results[0]},
                    commit=False)
                self.run.log({"avg_test_reward/STD_expert": expert_results[1]},
                             commit=False)

        self.run.finish()
        return dist
    
    
    def log_data(self, epoch, alpha, grad, dist, lr):

        # epoch
        self.run.log({"ep_logged": epoch}, commit=True)
        feat_keys = [f"feat_{i}" for i in range(alpha.shape[0])]
        # weight
        for ind, feat in enumerate(list(alpha)):
            self.run.log({f"alpha/{feat_keys[ind]}": feat}, commit=False)
        # grad
        for ind, feat in enumerate(list(grad)):
            self.run.log({f"grad/{feat_keys[ind]}": feat}, commit=False)
        # dist
        self.run.log({"dist/wasserstein_distance": dist}, commit=False)
        # lr
        self.run.log({"lr/lr": lr}, commit=False)


    def configure_experiment(self):
        # start wandb, create folders
        if self.cfg["fast"]:
            self.set_simple_params()

        # use wandb
        track = self.cfg["track"]
        group_name = self.cfg["group_name"]
        env_name = self.cfg["env_name"]
        self.exp_name = self.cfg["exp_name"]
        if track:
            wandb_project_name = self.cfg["wandb_project_name"]
            wandb_entity = self.cfg["wandb_entity"]
            notes = self.cfg["notes"]
            self.run = wandb.init(
                project=wandb_project_name,
                # entity=wandb_entity,
                # reinit=False,
                group=group_name,
                sync_tensorboard=self.cfg["sync_tb"],
                save_code=True,
                # job_type='outer_loop',
                # dir=log_path,
                # name=f"{exp_n}",
                notes=f"{notes}",
                config=self.cfg,
            )

            self.exp_name = f"{env_name}___group_{group_name}/run_{wandb.run.name}"
            # update configs to use elsewhere
            self.cfg["exp_name"] = self.exp_name
            self.cfg["tensorboard_log"] = f"checkpoints/{self.exp_name}/logs/"

        self.cfg["hydra_config"] = HydraConfig.get()
        self.cfg["results_path"] = HydraConfig.get().sweep.dir

        # create folder for files
        if not os.path.exists(f"checkpoints/{self.exp_name}/files/"):
            os.makedirs(f"checkpoints/{self.exp_name}/files/")

        print(
            f'\n\n ---- Started ... |{self.exp_name} | method - {self.cfg["feats_method"]}'
            f' | bc_only - {self.cfg["bc_only"]}\n')

    def save_experiment_params(self, objective, expert_res, agent_res):

        params = []
        with open(f"checkpoints/{self.exp_name}/experiment_params.csv",
                  "a") as fd:
            write = csv.writer(fd)

            params.append(str(objective))
            params.append(str(expert_res[0]))
            params.append(str(expert_res[1]))
            params.append(str(agent_res[0]))
            params.append(str(agent_res[1]))
            params.append(str(self.cfg["lr"]))
            params.append(str(self.cfg["gamma_feat"]))
            params.append(str(self.cfg["n_trajs"]))
            params.append(str(self.cfg["len_traj"]))
            params.append(str(self.cfg["epochs"]))
            params.append(str(self.cfg["d_states"]))
            params.append(str(self.cfg["total_timesteps"]))
            params.append(str(self.cfg["learning_rate"]))
            params.append(str(self.cfg["pi_size"]))
            params.append(str(self.cfg["samples_per_state"]))
            params.append(str(self.cfg["gamma"]))

            write.writerows([[
                "objective",
                "expert_mean",
                "expert_std",
                "agent_mean",
                "agent_std",
                "lr",
                "gamma_feat",
                "ntrajs",
                "len_traj",
                "epochs",
                "d_states",
                "total_timesteps",
                "learning_rate",
                "pi_size",
                "samples_per_state",
                "gamma",
            ]])

            write.writerows([params])

    def set_simple_params(self):
        self.cfg.device = "cpu"
        self.cfg.total_timesteps = 2000
        self.cfg.init_total_timesteps = 2000
        self.cfg.samples_per_state = 1
        self.cfg.n_trajs = 400
        self.cfg.len_traj = 100
        self.cfg.samples_per_state = 1
        self.cfg.epochs = 7
        self.cfg.save_freq = 3
        self.cfg.test_epoch = 3
        self.cfg.test_num = 2
        self.cfg.group_name = "fast_debug"

    def collect_rollouts(self, policy):
        nenvs = [self.create_env().envs[0] for _ in range(self.cfg["n_trajs"])]
        res = Parallel(n_jobs=self.cfg["n_threads"], prefer="threads")(
            delayed(generate_trajectories)(self.cfg, policy, nenvs[seed], seed)
            for seed in range(0, self.cfg["n_trajs"]))

        trs = [tr for (tr, id, r) in res]
        ids = [id for (tr, id, r) in res]
        rs = [r for (tr, id, r) in res]

        trajs = np.concatenate(trs)
        ts = np.concatenate(ids)
        rs = np.concatenate(rs)

        return trajs, ts, rs
