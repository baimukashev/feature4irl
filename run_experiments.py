'''
Run max-ent inverse RL experiment for continuous env.
'''
import os
os.environ['NUMEXPR_MAX_THREADS'] = '128'
os.environ["WANDB_SILENT"] = "true"
# os.environ.get("LD_LIBRARY_PATH", "")

import random
import numpy as np
import torch
import hydra
                                 


@hydra.main(config_path='feature4irl/cfg', config_name='exp_cfg', version_base=None)
def main(cfg):

    # TRY NOT TO MODIFY: seeding
    seed = cfg["seed"]  
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

        
    # load algo
    if cfg["algo_name"] == "maxent":
        from feature4irl.algorithms.maxent_irl import ContMaxEntIRL
        algo = ContMaxEntIRL(cfg=cfg)
    else:
        raise NotImplementedError
    
    return algo.train()
    

if __name__ == '__main__':
    main()