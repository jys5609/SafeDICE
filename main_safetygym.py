import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import time
import argparse
import os
import gym
import safety_gym
import pickle
from tqdm import tqdm
import getpass
import wandb

from algorithms.antidice import AntiDICE
from algorithms.dwbc import DWBC
from algorithms.cail import CAIL
from algorithms.bc import BC
from algorithms.dexperts import DExperts
import tensorflow as tf


np.set_printoptions(precision=4)

os.environ['WANDB_DIR'] = '/tmp/antidice/wandb'
os.environ['WANDB_API_KEY'] = '1666607857c706ed434217fca392e938aa80fcf4'

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--anti_ratio', type=float, default=0.0)
parser.add_argument('--algorithm', type=str, default='antidice') # antidice / bc / dwbc / dexperts / cail / expert_bc
# parser.add_argument('--env_name', type=str, default='maze2d') # maze2d / antmaze
parser.add_argument('--robot_name', type=str, default='Point') # Point / Car / Doggo
parser.add_argument('--task_name', type=str, default='Goal1') # Goal1 / Button1 / Push1
parser.add_argument('--num_unlabeled_expert_trajectory', type=int, default=1000) # cartpole: 1000 / walker: 5000
parser.add_argument('--num_unlabeled_anti_expert_trajectory', type=int, default=1000) # cartpole: 1000 / walker: 5000
parser.add_argument('--num_labeled_anti_expert_trajectory', type=int, default=50) # cartpole: 10 / walker: 50
parser.add_argument('--max_timesteps', type=int, default=1000)
parser.add_argument('--data_collection', type=bool, default=False)
parser.add_argument('--deterministic', type=bool, default=False)
parser.add_argument('--policy_type', type=str, default='TanhActor') # TanhMixtureActor / TanhActor
args = parser.parse_args()
config = vars(args)

print (args.algorithm)

alpha = args.alpha
num_unlabeled_expert_trajectory = args.num_unlabeled_expert_trajectory
num_unlabeled_anti_expert_trajectory = args.num_unlabeled_anti_expert_trajectory
num_labeled_anti_expert_trajectory = args.num_labeled_anti_expert_trajectory
num_total_anti_expert_trajectory = num_unlabeled_anti_expert_trajectory + num_labeled_anti_expert_trajectory

num_unlabeled_trajectory = num_unlabeled_expert_trajectory + num_unlabeled_anti_expert_trajectory
anti_ratio = num_unlabeled_anti_expert_trajectory / num_unlabeled_trajectory
max_timesteps = args.max_timesteps

algorithm = args.algorithm
eval_deterministic = args.deterministic
task_name = args.task_name
robot_name = args.robot_name
policy_type = args.policy_type

dataset_path = 'dataset/safetygym/'

seed = args.seed

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

if algorithm == 'antidice':
    import config.antidice_config as antidice_config
    config.update(antidice_config.hparams[0])
elif algorithm == 'dwbc':
    import config.dwbc_config as dwbc_config
    config.update(dwbc_config.hparams[0])
elif algorithm == 'cail':
    import config.cail_config as cail_config
    config.update(cail_config.hparams[0])
elif algorithm == 'dexperts':
    import config.dexperts_config as dexperts_config
    config.update(dexperts_config.hparams[0])
elif algorithm in ['bc', 'expert_bc']:
    import config.bc_config as bc_config
    config.update(bc_config.hparams[0])


np.random.seed(seed)
config['seed'] = seed
config['anti_ratio'] = anti_ratio
config['dataset_info'] = robot_name + '_' + task_name

# SafetyGym Env
env_name = 'Safexp-'+robot_name+task_name+'-v0'
vec_env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_name)] * 20)

def evaluate_vec_env(env, actor, num_episodes=500, deterministic=False):
    total_timesteps = 0
    all_returns = []
    all_costs = []

    num_iteration = int(num_episodes / env.num_envs)

    for i in range(num_iteration):
        state = env.reset()
        sum_return = np.zeros((env.num_envs,))
        sum_cost = np.zeros((env.num_envs,))
        for _ in range(max_timesteps):
            action = actor.step(state.astype(np.float32), deterministic=deterministic).numpy()
            next_state, reward, done, info = env.step(action)
            cost = np.array([info[i].get('cost', 0) for i in range(len(info))])

            total_timesteps += 1
            state = next_state
            sum_return += reward
            
            sum_cost += cost
        
        all_returns.append(sum_return)
        all_costs.append(sum_cost)

    all_returns = np.concatenate(all_returns).tolist()
    all_costs = np.concatenate(all_costs).tolist()

    mean_score = np.mean(all_returns)
    mean_cost = np.mean(all_costs)
    mean_timesteps = total_timesteps / num_episodes
    
    all_costs.sort()
    cvar_50 = np.mean(all_costs[-int(num_episodes*0.5):])
    cvar_20 = np.mean(all_costs[-int(num_episodes*0.2):])
    cvar_10 = np.mean(all_costs[-int(num_episodes*0.1):])
    cvar_5 = np.mean(all_costs[-int(num_episodes*0.05):])
    
    print(f'score / cost: {mean_score, mean_cost}')
    print(f': {mean_cost}')
    return mean_score, mean_cost, mean_timesteps, cvar_50, cvar_20, cvar_10, cvar_5


dict_keys = ['init_states', 'states', 'actions', 'next_states', 'costs', 'rewards', 'dones']

# Load anti_expert demonstrations (ppo)
with open('./dataset/safetygym/ppo_{}{}_s0.pickle'.format(robot_name, task_name), 'rb') as fr:
    anti_expert_demo = pickle.load(fr)
    anti_expert_init_states, anti_expert_states, \
    anti_expert_actions, anti_expert_next_states, \
    anti_expert_cost_violation, anti_expert_rewards, \
    anti_expert_dones = [anti_expert_demo[key][:num_unlabeled_anti_expert_trajectory*max_timesteps, :] for key in dict_keys]

    mean_anti_expert_costs = np.sum(anti_expert_cost_violation) / num_unlabeled_anti_expert_trajectory
    mean_anti_expert_rewards = np.sum(anti_expert_rewards) / num_unlabeled_anti_expert_trajectory
    config['mean_anti_expert_costs'] = mean_anti_expert_costs
    config['mean_anti_expert_rewards'] = mean_anti_expert_rewards

    labeled_anti_expert_init_states, labeled_anti_expert_states, \
    labeled_anti_expert_actions, labeled_anti_expert_next_states, \
    labeled_anti_expert_cost_violation, labeled_anti_expert_rewards, \
    labeled_anti_expert_dones = [anti_expert_demo[key][-num_labeled_anti_expert_trajectory*max_timesteps:, :] for key in dict_keys]

# Load expert demonstrations (ppo_lagrangian)
with open('./dataset/safetygym/ppo_lagrangian_{}{}_s0.pickle'.format(robot_name, task_name), 'rb') as fr:
    expert_demo = pickle.load(fr)
    expert_init_states, expert_states, \
    expert_actions, expert_next_states, \
    expert_cost_violation, expert_rewards, \
    expert_dones = [expert_demo[key][:num_unlabeled_expert_trajectory*max_timesteps, :] for key in dict_keys]    
    
    mean_expert_costs = np.sum(expert_cost_violation) / num_unlabeled_expert_trajectory
    mean_expert_rewards = np.sum(expert_rewards) / num_unlabeled_expert_trajectory
    config['mean_expert_costs'] = mean_expert_costs
    config['mean_expert_rewards'] = mean_expert_rewards
    
    unlabeled_init_states = np.concatenate([expert_init_states, anti_expert_init_states], axis=0).astype(np.float32)
    unlabeled_states = np.concatenate([expert_states, anti_expert_states], axis=0).astype(np.float32)
    unlabeled_actions = np.concatenate([expert_actions, anti_expert_actions], axis=0).astype(np.float32)
    unlabeled_next_states = np.concatenate([expert_next_states, anti_expert_next_states], axis=0).astype(np.float32)
    unlabled_cost_violation = np.concatenate([expert_cost_violation, anti_expert_cost_violation], axis=0).astype(np.float32)
    unlabeled_rewards = np.concatenate([expert_rewards, anti_expert_rewards], axis=0).astype(np.float32)   
    unlabeled_dones = np.concatenate([expert_dones, anti_expert_dones], axis=0).astype(np.float32)

    labeled_anti_expert_init_states = labeled_anti_expert_init_states.astype(np.float32)
    labeled_anti_expert_states = labeled_anti_expert_states.astype(np.float32)
    labeled_anti_expert_actions = labeled_anti_expert_actions.astype(np.float32)
    labeled_anti_expert_next_states = labeled_anti_expert_next_states.astype(np.float32)
    labeled_anti_expert_cost_violation = labeled_anti_expert_cost_violation.astype(np.float32)
    labeled_anti_expert_rewards = labeled_anti_expert_rewards.astype(np.float32)
    labeled_anti_expert_dones = labeled_anti_expert_dones.astype(np.float32)

print ('# of labeled anti-expert demonstrations: {}'.format(labeled_anti_expert_states.shape[0]))
print ('# of unlabeled demonstrations: {}'.format(unlabeled_states.shape[0]))

observation_dim = anti_expert_init_states.shape[-1]
is_discrete_action = anti_expert_actions[0].dtype == int
action_dim = anti_expert_actions.shape[-1]

if algorithm == 'antidice':
    if policy_type == 'TanhActor':
        imitator = AntiDICE(observation_dim, action_dim, mixture_actor=False, is_discrete_action=is_discrete_action, config=config)
    elif policy_type == 'TanhMixtureActor':
        imitator = AntiDICE(observation_dim, action_dim, mixture_actor=True, is_discrete_action=is_discrete_action, config=config)

elif algorithm == 'dwbc':
    if policy_type == 'TanhActor':
        imitator = DWBC(observation_dim, action_dim, mixture_actor=False, is_discrete_action=is_discrete_action, config=config)
    elif policy_type == 'TanhMixtureActor':
        imitator = DWBC(observation_dim, action_dim, mixture_actor=True, is_discrete_action=is_discrete_action, config=config)

elif algorithm == 'cail':
    if policy_type == 'TanhActor':
        imitator = CAIL(observation_dim, action_dim, mixture_actor=False, is_discrete_action=is_discrete_action, config=config)
    elif policy_type == 'TanhMixtureActor':
        imitator = CAIL(observation_dim, action_dim, mixture_actor=True, is_discrete_action=is_discrete_action, config=config)

elif algorithm == 'dexperts':
    if policy_type == 'TanhActor':
        imitator = DExperts(observation_dim, action_dim, mixture_actor=False, config=config)
    elif policy_type == 'TanhMixtureActor':
        imitator = DExperts(observation_dim, action_dim, mixture_actor=True, config=config)

elif algorithm in ['bc', 'expert_bc']:
    if policy_type == 'TanhActor':
        imitator = BC(observation_dim, action_dim, mixture_actor=False, config=config)
    elif policy_type == 'TanhMixtureActor':
        imitator = BC(observation_dim, action_dim, mixture_actor=True, config=config)

print("Save interval :", config['save_interval'])
# checkpoint dir
checkpoint_dir = './weights'
# os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filepath = f"{checkpoint_dir}/test.pickle"

training_info = {
            'wandb_run_id': None,
            'iteration': 0,
            'logs': [],
        }

# WANDB Logging
print("WANDB init...")
wandb_dir = os.environ.get('WANDB_DIR', 'wandb')
os.makedirs(wandb_dir, exist_ok=True)
wandb.init(project="safe_safetygym",
            entity="",
            dir=wandb_dir,
            config=config,
            save_code=True,
            resume=training_info['wandb_run_id'])
training_info['wandb_run_id'] = wandb.run.id

print(config['save_interval'])
config['total_iterations'] = config['total_iterations'] + 1

start_time = time.time()
pretrain_iter = 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def find_alpha(states, actions):
    data_num = states.shape[0]
    batch_size = config['batch_size']
    min_alpha = 1.0
    
    for i in range(int(data_num/ batch_size)):
        cost = imitator.cost(tf.concat([states[i*batch_size:(i+1)*batch_size], actions[i*batch_size:(i+1)*batch_size]], axis=-1))[0].numpy()
        cost = (1 / sigmoid(cost)) - 1
        alpha = np.min(cost)
        if min_alpha > alpha:
            min_alpha = alpha
    min_alpha -= 1e-5
    return min_alpha

if algorithm in ['antidice', 'dwbc']:
    # Pretrain discriminator for AntiDICE & DWBC
    while pretrain_iter < 100000: # 1000000:
        unlabeled_init_indices = np.random.randint(0, len(unlabeled_init_states), size=config['batch_size'])
        labeled_anti_expert_indices = np.random.randint(0, len(labeled_anti_expert_states), size=config['batch_size'])
        unlabeled_indices = np.random.randint(0, len(unlabeled_states), size=config['batch_size'])
        
        imitator.pretrain_discriminator(
                unlabeled_init_states[unlabeled_init_indices],
                labeled_anti_expert_states[labeled_anti_expert_indices],
                labeled_anti_expert_actions[labeled_anti_expert_indices],
                labeled_anti_expert_next_states[labeled_anti_expert_indices],
                unlabeled_states[unlabeled_indices],
                unlabeled_actions[unlabeled_indices],
                unlabeled_next_states[unlabeled_indices]
            )
        pretrain_iter += 1
        if pretrain_iter % 1000 == 0:
            print ('Pretrain iter: {}'.format(pretrain_iter))
    
    def reward_statistics(alpha):
        data_num = unlabeled_states.shape[0]
        batch_size = config['batch_size']
        reward_all = []
        r_mean, r_std = 0, 0
        for i in range(int(data_num/ batch_size)):        
            cost = imitator.cost(tf.concat([unlabeled_states[i*batch_size:(i+1)*batch_size], unlabeled_actions[i*batch_size:(i+1)*batch_size]], axis=-1))[0].numpy()
            cost = sigmoid(cost)
            
            reward = np.log(np.clip(1-(1+alpha)*cost, 1e-10, 1) / ((1-alpha)*(1-cost)))
            reward_all.append(reward)
        reward_all = np.concatenate(reward_all, axis=0)
        r_mean = np.mean(reward_all)
        r_std = np.std(reward_all)
        return r_mean, r_std
        
    def evaluate_discriminator():
        data_num = unlabeled_states.shape[0]
        batch_size = config['batch_size']
        positive = 0
        negative = 0
        
        for i in range(int(data_num/ batch_size)):        
            cost = imitator.cost(tf.concat([unlabeled_states[i*batch_size:(i+1)*batch_size], unlabeled_actions[i*batch_size:(i+1)*batch_size]], axis=-1))[0].numpy()
            cost = sigmoid(cost)
            positive += np.sum(cost > 0.5)
            negative += np.sum(cost < 0.5)
        
        print ('Unlabeled: positive {} / negative {}'.format(positive, negative))
        
        expert_unlabeled_states = unlabeled_states[:num_unlabeled_expert_trajectory*1000]
        expert_unlabeled_actions = unlabeled_actions[:num_unlabeled_expert_trajectory*1000]
        data_num = expert_unlabeled_states.shape[0]
        batch_size = config['batch_size']
        positive = 0
        negative = 0
        
        for i in range(int(data_num/ batch_size)):        
            cost = imitator.cost(tf.concat([expert_unlabeled_states[i*batch_size:(i+1)*batch_size], expert_unlabeled_actions[i*batch_size:(i+1)*batch_size]], axis=-1))[0].numpy()
            cost = sigmoid(cost)
            positive += np.sum(cost > 0.5)
            negative += np.sum(cost < 0.5)
        
        print ('Expert Unlabeled: positive {} / negative {}'.format(positive, negative))
        
        anti_expert_unlabeled_states = unlabeled_states[-num_unlabeled_anti_expert_trajectory*1000:]
        anti_expert_unlabeled_actions = unlabeled_actions[-num_unlabeled_anti_expert_trajectory*1000:]
        data_num = anti_expert_unlabeled_states.shape[0]
        batch_size = config['batch_size']
        positive = 0
        negative = 0
        
        for i in range(int(data_num/ batch_size)):        
            cost = imitator.cost(tf.concat([anti_expert_unlabeled_states[i*batch_size:(i+1)*batch_size], anti_expert_unlabeled_actions[i*batch_size:(i+1)*batch_size]], axis=-1))[0].numpy()
            cost = sigmoid(cost)
            positive += np.sum(cost > 0.5)
            negative += np.sum(cost < 0.5)
        
        print ('Anti Expert Unlabeled: positive {} / negative {}'.format(positive, negative))
        
        data_num = expert_states.shape[0]
        batch_size = config['batch_size']
        positive = 0
        negative = 0
        
        for i in range(int(data_num/ batch_size)):        
            cost = imitator.cost(tf.concat([expert_states[i*batch_size:(i+1)*batch_size], expert_actions[i*batch_size:(i+1)*batch_size]], axis=-1))[0].numpy()
            cost = sigmoid(cost)
            positive += np.sum(cost > 0.5)
            negative += np.sum(cost < 0.5)
        
        print ('EXPERT: positive {} / negative {}'.format(positive, negative))
        
        data_num = labeled_anti_expert_states.shape[0]
        batch_size = config['batch_size']
        positive = 0
        negative = 0
        
        for i in range(int(data_num/ batch_size)):        
            cost = imitator.cost(tf.concat([labeled_anti_expert_states[i*batch_size:(i+1)*batch_size], labeled_anti_expert_actions[i*batch_size:(i+1)*batch_size]], axis=-1))[0].numpy()
            cost = sigmoid(cost)
            positive += np.sum(cost > 0.5)
            negative += np.sum(cost < 0.5)
        
        print ('Anti EXPERT: positive {} / negative {}'.format(positive, negative))
    
    evaluate_discriminator()
    if algorithm == 'antidice':
        alpha = find_alpha(unlabeled_states, unlabeled_actions)
        r_mean, r_std = reward_statistics(alpha)
        
with tqdm(total=config['total_iterations'], initial=training_info['iteration'], desc='', disable=os.environ.get("DISABLE_TQDM", False), ncols=70) as pbar:
    while training_info['iteration'] < config['total_iterations']:
        unlabeled_init_indices = np.random.randint(0, len(unlabeled_init_states), size=config['batch_size'])
        labeled_anti_expert_indices = np.random.randint(0, len(labeled_anti_expert_states), size=config['batch_size'])
        unlabeled_indices = np.random.randint(0, len(unlabeled_states), size=config['batch_size'])
        expert_indices = np.random.randint(0, len(expert_states), size=config['batch_size'])

        if algorithm == 'antidice':
            # alpha = find_alpha(unlabeled_states[unlabeled_indices], unlabeled_actions[unlabeled_indices])
            
            info_dict = imitator.update(
                    unlabeled_init_states[unlabeled_init_indices],
                    labeled_anti_expert_states[labeled_anti_expert_indices],
                    labeled_anti_expert_actions[labeled_anti_expert_indices],
                    labeled_anti_expert_next_states[labeled_anti_expert_indices],
                    unlabeled_states[unlabeled_indices],
                    unlabeled_actions[unlabeled_indices],
                    unlabeled_next_states[unlabeled_indices], np.array(alpha, dtype=np.float32),
                    np.array(r_mean, dtype=np.float32), np.array(r_std, dtype=np.float32)
                )
        
        elif algorithm in ['dwbc', 'cail']:
            info_dict = imitator.update(
                    unlabeled_init_states[unlabeled_init_indices],
                    labeled_anti_expert_states[labeled_anti_expert_indices],
                    labeled_anti_expert_actions[labeled_anti_expert_indices],
                    labeled_anti_expert_next_states[labeled_anti_expert_indices],
                    unlabeled_states[unlabeled_indices],
                    unlabeled_actions[unlabeled_indices],
                    unlabeled_next_states[unlabeled_indices]
                )
        
        elif algorithm == 'dexperts':
            alpha = config['alpha']
            info_dict = imitator.update(
                    unlabeled_states[unlabeled_indices], 
                    unlabeled_actions[unlabeled_indices], 
                    labeled_anti_expert_states[labeled_anti_expert_indices],
                    labeled_anti_expert_actions[labeled_anti_expert_indices], np.array(alpha, dtype=np.float32)
                )
        
        elif algorithm == 'bc':
            info_dict = imitator.update(unlabeled_states[unlabeled_indices], unlabeled_actions[unlabeled_indices])
        
        elif algorithm == 'expert_bc':
            info_dict = imitator.update(expert_states[expert_indices], expert_actions[expert_indices])
        
        elif algorithm == 'optidice':
            pass

        if training_info['iteration'] % config['log_interval'] == 0:
            info_dict.update({'alpha': alpha})
            
            start = time.time()
            
            average_returns, average_costs, evaluation_timesteps, cvar_50, cvar_20, cvar_10, cvar_5 = evaluate_vec_env(vec_env, imitator, deterministic=eval_deterministic)

            end = time.time()
            print ('vector env time: ', end-start)
            
            info_dict.update({'eval': average_returns, 'cost': average_costs, 
                              'CVar50': cvar_50, 'CVar20': cvar_20, 'CVar10': cvar_10, 'CVar5': cvar_5})
            
            print(f'Eval: ave returns=d: {average_returns}'
                  f'Cost: ave costs=d: {average_costs}'
                    f' ave episode length={evaluation_timesteps}'
                    f' / elapsed_time={time.time() - start_time} ({training_info["iteration"] / (time.time() - start_time)} it/sec)')
            print('=========================')
            for key, val in info_dict.items():
                print(f'{key:25}: {val:8.3f}')
            print('=========================')

            wandb.log(info_dict, step=training_info['iteration'])
            training_info['logs'].append({'step': training_info['iteration'], 'log': info_dict})
            print(f'timestep {training_info["iteration"]} - log update...')
            print('Done!', flush=True)
            
        if training_info['iteration'] % config['save_interval'] == 0 and training_info['iteration'] > 0:
            imitator.save(checkpoint_filepath, training_info)

        training_info['iteration'] += 1
        pbar.update(1)

