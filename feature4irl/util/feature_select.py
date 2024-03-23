import numpy as np
from itertools import combinations_with_replacement
from sklearn.preprocessing import RobustScaler

def select_feat_extractor(env_name, states, cfg):

    if env_name == 'Pendulum-v1':
        feature_expectations = pend_feat_extract(states, cfg)
        
    elif env_name== 'CartPole-v1':
        feature_expectations = cart_feat_extract(states, cfg)

    elif env_name== 'Acrobot-v1':
        feature_expectations = acrobot_feat_extract(states, cfg)
                
    # elif env_name== 'Reacher-v4':
    #     feature_expectations = reacher_feat_extract(states, cfg)

    # elif env_name== 'Hopper-v4':
    #     feature_expectations = hopper_feat_extract(states, cfg)
                
    # elif env_name== 'highway-fast-v0':
    #     feature_expectations = high_feat_extract(states, cfg)
    
    # elif env_name== 'LunarLander-v2':
    #     feature_expectations = lander_feat_extract(states, cfg)
    
    # elif env_name== 'Swimmer-v4':
    #     feature_expectations = hopper_feat_extract(states, cfg)

    # elif env_name== 'InvertedPendulum-v4':
    #     feature_expectations = hopper_feat_extract(states, cfg)
        
    else:
        raise NotImplementedError
    
    return feature_expectations




# ██████  ███████ ███    ██ ██████  ██    ██ ██      ██    ██ ███    ███ 
# ██   ██ ██      ████   ██ ██   ██ ██    ██ ██      ██    ██ ████  ████ 
# ██████  █████   ██ ██  ██ ██   ██ ██    ██ ██      ██    ██ ██ ████ ██ 
# ██      ██      ██  ██ ██ ██   ██ ██    ██ ██      ██    ██ ██  ██  ██ 
# ██      ███████ ██   ████ ██████   ██████  ███████  ██████  ██      ██ 


def pend_feat_extract(states, cfg):
    # Given pendulum 
    # https://gymnasium.farama.org/environments/classic_control/pendulum/
    
    #     angle = np.arctan2(y, x) / 3.15
    
    use_norm = cfg["normalize_feats"]
     
    if use_norm:
        # read observaion values and normalize
        mu =  np.array([0.7473041415214539, -0.04867269843816757, 0.11523590981960297])
        std = np.array([0.5509276390075684, 0.36831530928611755, 1.6884348392486572])
        states_std = (states - mu) / std
        
    else:
        states_std = states
            
    # select subset
    feat_selection = cfg["feats_method"]
    feats = []
    
    if feat_selection == 'first':
        feats.append(states_std[0])
        feats.append(states_std[1])    
        feats.append(states_std[2])
        
    elif feat_selection == 'all':
        
        curr_point = states_std
        
        # find features
        second_degree = [x * y for x, y in combinations_with_replacement(curr_point, 2)]

        # Combine both first and second degree polynomials
        feats = curr_point.tolist() + second_degree
        
    elif feat_selection == 'random':
        
        curr_point = states_std
        
        # find features
        second_degree = [x * y for x, y in combinations_with_replacement(curr_point, 2)]

        # Combine both first and second degree polynomials
        feats_list = curr_point.tolist() + second_degree
        
        # [6, 3, 0, 4]
        feats.append(feats_list[5])
        feats.append(feats_list[2])    
        feats.append(feats_list[3])
            
    elif feat_selection == 'manual':
        
        # manual        
        feats.append(states_std[0] ** 2)
        feats.append(states_std[1] ** 2)    
        feats.append(states_std[2] ** 2)

    elif feat_selection == 'proposed':

        # proposed -- f1f3 f3f3 f1f1
        feats.append(states_std[0])
        feats.append(states_std[1] ** 2)    
        feats.append(states_std[2] ** 2)
        
    elif feat_selection == 'other':
        pass
    
    else:
        NotImplementedError()

    return np.array(feats)
    


#  ██████  █████  ██████  ████████ ██████   ██████  ██      ███████ 
# ██      ██   ██ ██   ██    ██    ██   ██ ██    ██ ██      ██      
# ██      ███████ ██████     ██    ██████  ██    ██ ██      █████   
# ██      ██   ██ ██   ██    ██    ██      ██    ██ ██      ██      
#  ██████ ██   ██ ██   ██    ██    ██       ██████  ███████ ███████ 

                                                                                                                               
def cart_feat_extract(states, cfg):

    use_norm = cfg["normalize_feats"]
     
    if use_norm:
        # read observaion values and normalize
        mu =  np.array([-0.4351567029953003, -0.20982103049755096, -0.003578165778890252, 0.03000437282025814])
        std = np.array([0.4223495125770569, 0.5075123310089111, 0.06706511229276657, 0.35969388484954834])
        states_std = (states - mu) / std
        
    else:
        states_std = states
            
    # select subset
    feat_selection = cfg["feats_method"]
    feats = []
    
    if feat_selection == 'first':
        feats.append(states_std[0])
        feats.append(states_std[1])    
        feats.append(states_std[2])
        feats.append(states_std[3])
        
    elif feat_selection == 'all':
        
        curr_point = states_std
        
        # find features
        second_degree = [x * y for x, y in combinations_with_replacement(curr_point, 2)]

        # Combine both first and second degree polynomials
        feats = curr_point.tolist() + second_degree
        
    elif feat_selection == 'random':
        
        curr_point = states_std
        
        # find features
        second_degree = [x * y for x, y in combinations_with_replacement(curr_point, 2)]

        # Combine both first and second degree polynomials
        feats_list = curr_point.tolist() + second_degree
        
        # [6, 3, 0, 4]
        feats.append(feats_list[6])
        feats.append(feats_list[3])    
        feats.append(feats_list[0])
        feats.append(feats_list[4])
            
    elif feat_selection == 'manual':
        
        # manual        
        feats.append(states_std[0] ** 2)
        feats.append(states_std[1] * states_std[1])
        feats.append(states_std[2] * states_std[2])
        feats.append(states_std[3] * states_std[3])

    elif feat_selection == 'proposed':

        # proposed -- f1f3 f3f3 f1f1
        feats.append(states_std[1] * states_std[1])
        feats.append(states_std[1] * states_std[3])    
        feats.append(states_std[3] * states_std[3])
        feats.append(states_std[2] * states_std[2])
        
    elif feat_selection == 'other':
        pass
    
    else:
        NotImplementedError()

    return np.array(feats)


#  █████   ██████ ██████   ██████  ██████   ██████  ████████ 
# ██   ██ ██      ██   ██ ██    ██ ██   ██ ██    ██    ██    
# ███████ ██      ██████  ██    ██ ██████  ██    ██    ██    
# ██   ██ ██      ██   ██ ██    ██ ██   ██ ██    ██    ██    
# ██   ██  ██████ ██   ██  ██████  ██████   ██████     ██    
                                                           
                                                           
def acrobot_feat_extract(states, cfg):

    # features    
    use_norm = cfg["normalize_feats"]
     
    if use_norm:
        # read observaion values and normalize
        mu =  np.array([0.5131067037582397, 0.0005170440417714417, 0.1857973039150238, 0.0010799641022458673, -0.029419755563139915, 0.7311022877693176])
        std = np.array([0.5434738993644714, 0.6643465757369995, 0.6896401643753052, 0.6999107599258423, 2.411879062652588, 4.213535785675049])
        states_std = (states - mu) / std
        
    else:
        states_std = states
            
    # select subset
    feat_selection = cfg["feats_method"]
    feats = []
    
    if feat_selection == 'first':
        feats.append(states_std[0])
        feats.append(states_std[1])    
        feats.append(states_std[2])
        feats.append(states_std[3])
        feats.append(states_std[4])
        feats.append(states_std[5])
        
    elif feat_selection == 'all':
        
        curr_point = states_std
        # find features
        second_degree = [x * y for x, y in combinations_with_replacement(curr_point, 2)]
        # Combine both first and second degree polynomials
        feats = curr_point.tolist() + second_degree
        
    elif feat_selection == 'random':
        
        curr_point = states_std
        # find features
        second_degree = [x * y for x, y in combinations_with_replacement(curr_point, 2)]
        # Combine both first and second degree polynomials
        feats_list = curr_point.tolist() + second_degree
        
        # 21, 7, 12, 19
        feats.append(feats_list[21])
        feats.append(feats_list[7])    
        feats.append(feats_list[12])
        feats.append(feats_list[19])
            
    elif feat_selection == 'manual':
        # manual        
        feats.append(states_std[0] * states_std[2] - states_std[1] * states_std[3])
        feats.append(states_std[0])
        feats.append(states_std[2] ** 2)
        feats.append(states_std[1] ** 2)

    elif feat_selection == 'proposed':
        # proposed
        feats.append(states_std[2] ** 2)
        feats.append(states_std[3] ** 2)
        feats.append(states_std[5] ** 2)
        
        
    elif feat_selection == 'other':
        pass
    
    else:
        NotImplementedError()

            
    return np.array(feats)



