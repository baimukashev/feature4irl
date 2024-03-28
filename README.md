# Automated Feature Selection for Inverse Reinforcement Learning

## Overview
This repository contains the source code and data for the paper [Automated Feature Selection for Inverse Reinforcement Learning](http://arxiv.org/abs/2403.15079)

## Installation

* Clone the repo `git clone git@github.com:baimukashev/feature4irl.git`
* Run `cd feature4irl`
* Run `pip install -r requirements.txt`
* Optionally, setup [Weight and Biases](https://docs.wandb.ai/quickstart) for tracking the experiments.

## How to run the code
* Run `./experiments_{acr, pend, cart}.sh` for training all baselines for each task respectively.
* Results of the experiments including the simulation videos, data will be then saved in `./checkpoints` folder.

## Configuration
Configuration files for each environment are located in ```feature4irl/cfg/```.
