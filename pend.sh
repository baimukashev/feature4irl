#!/bin/bash

# Start a new tmux session
tmux new-session -d -s pend

# Split the tmux window vertically into two panes
tmux send-keys -t pend:0.0 'export CUDA_VISIBLE_DEVICES=0' C-m
tmux send-keys -t pend:0.0 'conda activate irl_env' C-m

tmux send-keys -t pend:0.0 'python run_experiments.py\
                                    env_agent_algo=pend_ppo_maxent\
                                    group_name=all_exp1\
                                    notes='all states'\
                                    d_states=9\
                                    lr=0.2\
                                    seed=0\
                                    normalize_feats=False\
                                    feats_method=all' C-m

tmux split-window -h
tmux send-keys -t pend:0.1 'export CUDA_VISIBLE_DEVICES=1' C-m
tmux send-keys -t pend:0.1 'conda activate irl_env' C-m

tmux send-keys -t pend:0.1 'python run_experiments.py\
                                    env_agent_algo=pend_ppo_maxent\
                                    group_name=run_multi_exp2\
                                    notes='first order states'\
                                    d_states=3\
                                    lr=0.2\
                                    normalize_feats=False\
                                    feats_method=first' C-m

tmux split-window -h
tmux send-keys -t pend:0.2 'export CUDA_VISIBLE_DEVICES=0' C-m
tmux send-keys -t pend:0.2 'conda activate irl_env' C-m

tmux send-keys -t pend:0.2 'python run_experiments.py\
                                    env_agent_algo=pend_ppo_maxent\
                                    group_name=run_multi_exp3\
                                    notes='random feats'\
                                    d_states=3\
                                    lr=0.2\
                                    normalize_feats=False\
                                    feats_method=random' C-m


tmux send-keys -t pend:0.3 'export CUDA_VISIBLE_DEVICES=1' C-m
tmux send-keys -t pend:0.3 'conda activate irl_env' C-m

tmux split-window -h
tmux send-keys -t pend:0.3 'python run_experiments.py\
                                    env_agent_algo=pend_ppo_maxent\
                                    group_name=run_multi_exp4\
                                    notes='proposed'\
                                    d_states=3\
                                    lr=0.2\
                                    normalize_feats=False\
                                    feats_method=proposed' C-m


tmux split-window -h

tmux send-keys -t pend:0.4 'export CUDA_VISIBLE_DEVICES=1' C-m
tmux send-keys -t pend:0.4 'conda activate irl_env' C-m

tmux send-keys -t pend:0.4 'python run_experiments.py\
                                    env_agent_algo=pend_ppo_maxent\
                                    group_name=run_multi_exp5\
                                    notes='manual'\
                                    d_states=3\
                                    lr=0.2\
                                    normalize_feats=False\
                                    feats_method=manual' C-m

