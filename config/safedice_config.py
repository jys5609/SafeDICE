from itertools import product
from copy import deepcopy

grid_search = {
    "algorithm":[
        "safedice",
    ],
    # "seed": [0],
    "env_id": [
        "maze2d"
        #"NoisyHopper-0.0-v2",
    ],
    "weight_normalization":[
        "self",         # self normalization
    ],
    "use_last_layer_bias_cost": [
        False,
        # True,
    ],
    "use_last_layer_bias_critic": [
        False,
        # True,
    ],
    "grad_reg_coeffs":[
        (10, 1e-6),
    ],
    # "alpha":[
    #     0.1
    # ],
    "actor_type":[
        "tanh-normal",
        # "normal",
    ],
    # "imperfect_dataset_info": [  # dataset_name, num_traj
    #     (["expert-v0", "medium-v0", "random-v0"], [100, 1600, 1600]),
    # ],
    # "expert_dataset_info": [  # dataset_name, num_traj
    #     ("expert-v0", 30)
    #     # ("expert-v2", 30)  # <- for vanilla Hopper,Walker2d,...
    # ],
    "total_iterations": [
        int(1e6) # cartpole: 
    ],
    "log_interval":[
        int(1e4),
    ],
    "batch_size": [
        512,
    ],
    "hidden_size":[
        256,
    ],
    "cost_lr":[
        1e-4,
        # 3e-5
    ],
    "critic_lr":[
        1e-4,
        # 3e-5
    ],
    "actor_lr":[
        # 1e-4,
        1e-5
    ],
    "gamma":[
        0.99,
    ],
    "save_interval":[
        50000,
    ],
    "kernel_initializer":[
        "he_normal",
        # "he_uniform",
        # "glorot_uniform",
        # "glorot_normal"
    ],
    "state_matching": [
        False,
    ],
    "closed_form_mu": [
        True,
    ],
    "resume": [
        False,
    ],
}

hparams = []

for grid_search_value in product(*grid_search.values()):
    grid_dict = dict(zip(grid_search.keys(), grid_search_value))

    hparam = deepcopy(grid_dict)
    hparams.append(hparam)

if __name__ == '__main__':
    for pid, hparam in enumerate(hparams):
        print(f'{pid:3d}: {hparam}')