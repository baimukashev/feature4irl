#!/bin/bash

# RUN all experiments

python run_experiments.py \
  env_agent_algo=cart_ppo_maxent \
  group_name=all_exp \
  notes='all states' \
  d_states=14 \
  lr=0.2 \
  normalize_feats=False \
  feats_method=all

python run_experiments.py \
  env_agent_algo=cart_ppo_maxent \
  group_name=first_exp \
  notes='first order states' \
  d_states=4 \
  lr=0.2 \
  normalize_feats=False \
  feats_method=first

python run_experiments.py \
  env_agent_algo=cart_ppo_maxent \
  group_name=random_exp \
  notes='random feats' \
  d_states=4 \
  lr=0.2 \
  normalize_feats=False \
  feats_method=random

python run_experiments.py \
  env_agent_algo=cart_ppo_maxent \
  group_name=proposed_exp \
  notes='proposed' \
  d_states=4 \
  lr=0.2 \
  normalize_feats=False \
  feats_method=proposed

python run_experiments.py \
  env_agent_algo=cart_ppo_maxent \
  group_name=manual_exp \
  notes='manual' \
  d_states=4 \
  lr=0.2 \
  normalize_feats=False \
  feats_method=manual

