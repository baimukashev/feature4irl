import torch
import numpy as np
import gymnasium as gym


class Agent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def act(self, obs):
        pass

    def learn(self, obs):
        pass

    def save(self, obs):
        pass

    def test(self, obs):
        pass

    def load(self, obs):
        pass

    def save_render(
        self,
        video_dir,
        test_num=10,
        duration=200,
        seed=100000,
        test_env=None,
        policy=None,
    ):

        rewards = []
        model = self.policy
        all_feats = []

        for idx in range(self.cfg["test_num"]):

            env = test_env.envs[0]

            if self.cfg["render"]:

                print("Testing....")
                env.unwrapped.render_mode = "rgb_array"
                # env.metadata["render_fps"] = 30

                env.metadata["offscreen_rendering"] = True

                video_path = f"{video_dir}_videos/seed_{seed}_+_{idx}"

                env = gym.wrappers.RecordVideo(
                    env, video_path, disable_logger=True, episode_trigger=lambda e: True
                )

            ind = 0
            done = False
            terminated = False
            episode_reward = 0

            obs, _ = env.reset(seed=seed + idx)
            dur = self.cfg["len_traj"]

            feats = []

            while ind < dur and not done and not terminated:

                action, *_ = model.predict(torch.tensor(obs))

                obs, reward, done, terminated, info = env.step(action)

                episode_reward += reward
                ind += 1

            rewards.append(episode_reward)

            all_feats.append(np.array(feats))

            env.close()

        return (np.mean(rewards), np.std(rewards))
