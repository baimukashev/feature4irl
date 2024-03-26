import torch
import numpy as np
import gymnasium as gym
import copy
from .feature_select import select_feat_extractor


def generate_trajectories(cfg, policy, env, seed):

    # env = copy.deepcopy(env)
    samples_per_initial_state = cfg["samples_per_state"]

    trajectories_obs = []
    trajectories_steps = []
    trajectories_rewards = []

    for _ in range(1):

        i = 0
        done = 0
        trunc = 0
        obs, _ = env.reset(seed=seed)

        while i < cfg["len_traj"] and not done and not trunc:
            action, *_ = policy.predict(torch.tensor(obs).float(), deterministic=False)
            next_obs, reward, done, trunc, info = env.step(action)

            trajectories_obs.append(next_obs)
            trajectories_steps.append(i)
            trajectories_rewards.append(reward)

            obs = next_obs

            i += 1

    env.close()

    return (
        np.array(trajectories_obs),
        np.array(trajectories_steps),
        np.array(trajectories_rewards),
    )


def find_feature_expectations(cfg, trajectories, steps):

    gamma = cfg["gamma_feat"]
    feature_expectations = np.zeros(cfg["d_states"])
    env_name = cfg["env_name"]

    for i, states in enumerate(trajectories):

        # import pdb; pdb.set_trace()
        # select feature extractor
        features = select_feat_extractor(env_name, states, cfg)  # phi(s)
        features_discounted = features * (gamma ** steps[i])  # phi(s) * (gamma ** time)

        feature_expectations += (
            features_discounted  # phi_exp += phi(s) * (gamma ** time)
        )

    feature_expectations /= trajectories.shape[0]

    return feature_expectations
