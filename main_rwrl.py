import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import time
import argparse
import os
import gym
import pickle
from tqdm import tqdm
import getpass
import wandb

from algorithms.safedice import SafeDICE
from algorithms.dwbc import DWBC
from algorithms.cail import CAIL
from algorithms.bc import BC
from algorithms.dexperts import DExperts
from realworldrl_suite.environments.cartpole import slider_pos_constraint
from realworldrl_suite.environments.walker import joint_velocity_constraint
from realworldrl_suite.environments.quadruped import joint_angle_constraint as joint_angle_constraint_quadruped
from realworldrl_suite.environments.humanoid import joint_angle_constraint as joint_angle_constraint_humanoid
import collections
import tensorflow as tf

from absl import app
from absl import flags
import realworldrl_suite.environments as rwrl
import dm2gym.envs.dm_suite_env as dm2gym
# from baselines import bench


np.set_printoptions(precision=4)

os.environ['WANDB_DIR'] = '/tmp/safedice/wandb'
# os.environ['WANDB_API_KEY'] = '' 

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--anti_ratio', type=float, default=0.0)
parser.add_argument('--algorithm', type=str, default='safedice') # safedice / bc / dwbc / dexperts / cail / expert_bc
# parser.add_argument('--env_name', type=str, default='maze2d') # maze2d / antmaze
parser.add_argument('--domain_name', type=str, default='walker') # cartpole / walker / quadruped / humanoid
parser.add_argument('--task_name', type=str, default='realworld_walk') # realworld_swingup / realworld_walk / realworld_walk / realworld_walk
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
domain_name = args.domain_name
task_name = args.task_name
policy_type = args.policy_type

seed = args.seed

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

if algorithm == 'safedice':
    import config.safedice_config as safedice_config
    config.update(safedice_config.hparams[0])
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
config['dataset_info'] = domain_name + '_' + task_name

class GymEnv(dm2gym.DMSuiteEnv):
  """Wrapper that convert a realworldrl environment to a gym environment."""

  def __init__(self, env):
    """Constructor. We reuse the facilities from dm2gym."""
    self.env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': round(1. / self.env.control_timestep())
    }
    self.observation_space = dm2gym.convert_dm_control_to_gym_space(
        self.env.observation_spec())
    self.action_space = dm2gym.convert_dm_control_to_gym_space(
        self.env.action_spec())
    self.viewer = None

constraints_dict = {'cartpole': collections.OrderedDict([('slider_pos_constraint', slider_pos_constraint)]),
                    'walker': collections.OrderedDict([('joint_velocity_constraint', joint_velocity_constraint)]),
                    'quadruped': collections.OrderedDict([('joint_angle_constraint', joint_angle_constraint_quadruped)])
                    }

safety_coeff_dict = {'cartpole': 0.2,
                    'walker': 0.2,
                    'quadruped': 0.35
                    }

raw_env = rwrl.load(domain_name=domain_name,
                    task_name=task_name,
                    safety_spec={'enable': True, 
                                 'observations': True, 
                     'safety_coeff': safety_coeff_dict[domain_name],
                     'constraints': constraints_dict[domain_name]
                     },
                    log_output=os.path.join('./', 'log.npz'),
                    environment_kwargs=dict(
                        log_safety_vars=True, log_every=20, flat_observation=True))

env = GymEnv(raw_env)

vec_env = gym.vector.AsyncVectorEnv([lambda: GymEnv(rwrl.load(domain_name=domain_name,
                    task_name=task_name,
                    safety_spec={'enable': True, 
                                 'observations': True, 
                     'safety_coeff': safety_coeff_dict[domain_name],
                     'constraints': constraints_dict[domain_name]
                     },
                    log_output=os.path.join('./', 'log.npz'),
                    environment_kwargs=dict(
                        log_safety_vars=True, log_every=20, flat_observation=True)))] * 100)


def evaluate_vec_env(env, actor, num_episodes=500, deterministic=False, cost_violation_threshold=0, strict_cost_violation_threshold=0):
    total_timesteps = 0
    all_returns = []
    all_costs = []

    num_iteration = int(num_episodes / env.num_envs)

    for i in range(num_iteration):
        state = env.reset()
        sum_return = np.zeros((env.num_envs,))
        sum_cost = np.zeros((env.num_envs,))
        done = False
        for _ in range(max_timesteps):
            action = actor.step(state[:, :-1].astype(np.float32), deterministic=deterministic).numpy()
            next_state, reward, done, _ = env.step(action)

            total_timesteps += 1
            state = next_state
            sum_return += reward
            sum_cost += (1 - next_state[:, -1])
        
        all_returns.append(sum_return)
        all_costs.append(sum_cost)

    all_returns = np.concatenate(all_returns).tolist()
    all_costs = np.concatenate(all_costs).tolist()

    mean_score = np.mean(all_returns)
    mean_cost = np.mean(all_costs)
    mean_timesteps = total_timesteps / num_episodes
    cost_violation_ratio = np.sum(np.array(all_costs) > cost_violation_threshold) / num_episodes
    strict_cost_violation_ratio = np.sum(np.array(all_costs) > strict_cost_violation_threshold) / num_episodes
    
    all_costs.sort()
    cvar_50 = np.mean(all_costs[-int(num_episodes*0.5):])
    cvar_20 = np.mean(all_costs[-int(num_episodes*0.2):])
    cvar_10 = np.mean(all_costs[-int(num_episodes*0.1):])
    cvar_5 = np.mean(all_costs[-int(num_episodes*0.05):])
    
    print(f'score / cost: {mean_score, mean_cost}')
    print(f': {mean_cost}')
    print ('cost violation ratio: {}'.format(cost_violation_ratio))
    return mean_score, mean_cost, mean_timesteps, cost_violation_ratio, strict_cost_violation_ratio, cvar_50, cvar_20, cvar_10, cvar_5


def evaluate(env, actor, num_episodes=100, deterministic=False, cost_violation_threshold=0, strict_cost_violation_threshold=0):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      train_env_id: train_env_id to compute normalized score
      num_episodes: A number of episodes to average the policy on.
      deterministic: whether deterministic or stochastic action
    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = 0
    total_returns = 0
    total_costs = 0
    all_returns = []
    all_costs = []

    for i in range(num_episodes):
        state = env.reset()
        sum_return = 0
        sum_cost = 0
        done = False
        for _ in range(max_timesteps):
            action = actor.step(state[:-1], deterministic=deterministic)[0].numpy()

            next_state, reward, done, _ = env.step(action)

            total_returns += reward
            total_costs += (1 - next_state[-1])
            total_timesteps += 1
            state = next_state

            sum_return += reward
            sum_cost += (1 - next_state[-1])

        all_returns.append(sum_return)
        all_costs.append(sum_cost)

    mean_score = total_returns / num_episodes
    mean_cost = total_costs / num_episodes
    mean_timesteps = total_timesteps / num_episodes
    cost_violation_ratio = np.sum(np.array(all_costs) > cost_violation_threshold) / num_episodes
    strict_cost_violation_ratio = np.sum(np.array(all_costs) > strict_cost_violation_threshold) / num_episodes
    
    all_costs.sort()
    cvar_50 = np.mean(all_costs[-int(num_episodes*0.5):])
    cvar_20 = np.mean(all_costs[-int(num_episodes*0.2):])
    cvar_10 = np.mean(all_costs[-int(num_episodes*0.1):])
    cvar_5 = np.mean(all_costs[-int(num_episodes*0.05):])
    
    print(f'score / cost: {mean_score, mean_cost}')
    print(f': {mean_cost}')
    print ('cost violation ratio: {}'.format(cost_violation_ratio))
    return mean_score, mean_cost, mean_timesteps, cost_violation_ratio, strict_cost_violation_ratio, cvar_50, cvar_20, cvar_10, cvar_5


dict_keys = ['init_states', 'states', 'actions', 'next_states', 'cost_violation', 'rewards', 'dones']

with open('./dataset/rwrl/{}/anti_expert_6000_1000_0.2.pickle'.format(domain_name), 'rb') as fr: # for walker
    anti_expert_demo = pickle.load(fr)
    anti_expert_init_states, anti_expert_states, \
    anti_expert_actions, anti_expert_next_states, \
    anti_expert_cost_violation, anti_expert_rewards, \
    anti_expert_dones = [anti_expert_demo[key][:num_unlabeled_anti_expert_trajectory*max_timesteps, :] for key in dict_keys]

    mean_anti_expert_costs = np.sum(anti_expert_cost_violation) / num_unlabeled_anti_expert_trajectory
    config['mean_anti_expert_costs'] = mean_anti_expert_costs

    labeled_anti_expert_init_states, labeled_anti_expert_states, \
    labeled_anti_expert_actions, labeled_anti_expert_next_states, \
    labeled_anti_expert_cost_violation, labeled_anti_expert_rewards, \
    labeled_anti_expert_dones = [anti_expert_demo[key][-num_labeled_anti_expert_trajectory*max_timesteps:, :] for key in dict_keys]

with open('./dataset/rwrl/{}/expert_6000_1000_0.2.pickle'.format(domain_name), 'rb') as fr: # for walker
    expert_demo = pickle.load(fr)
    expert_init_states, expert_states, \
    expert_actions, expert_next_states, \
    expert_cost_violation, expert_rewards, \
    expert_dones = [expert_demo[key][:num_unlabeled_expert_trajectory*max_timesteps, :] for key in dict_keys]    
    
    # cost_violation_threshold = np.max(expert_costs)
    mean_expert_costs = np.sum(expert_cost_violation) / num_unlabeled_expert_trajectory
    config['mean_expert_costs'] = mean_expert_costs
    
    mean_expert_costs_std = np.std(np.sum(np.reshape(expert_cost_violation, (max_timesteps, num_unlabeled_expert_trajectory, 1)), axis=0))
    strict_cost_violation_threshold = mean_expert_costs + 1.96 * mean_expert_costs_std / np.sqrt(num_unlabeled_expert_trajectory)
    
    cost_violation_threshold = (mean_expert_costs + mean_anti_expert_costs) / 2
    # strict_cost_violation_threshold = mean_anti_expert_costs - 100
    config['cost_violation_threshold'] = cost_violation_threshold
    config['strict_cost_violation_threshold'] = strict_cost_violation_threshold
    
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

if algorithm == 'safedice':
    if policy_type == 'TanhActor':
        imitator = SafeDICE(observation_dim, action_dim, mixture_actor=False, is_discrete_action=is_discrete_action, config=config)
    elif policy_type == 'TanhMixtureActor':
        imitator = SafeDICE(observation_dim, action_dim, mixture_actor=True, is_discrete_action=is_discrete_action, config=config)

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
wandb.init(project="safedice_{}".format(domain_name),
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



if algorithm in ['safedice', 'dwbc']:
    # Pretrain discriminator for SafeDICE & DWBC
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
    if algorithm == 'safedice':
        alpha = find_alpha(unlabeled_states, unlabeled_actions)
        
with tqdm(total=config['total_iterations'], initial=training_info['iteration'], desc='', disable=os.environ.get("DISABLE_TQDM", False), ncols=70) as pbar:
    while training_info['iteration'] < config['total_iterations']:
        unlabeled_init_indices = np.random.randint(0, len(unlabeled_init_states), size=config['batch_size'])
        labeled_anti_expert_indices = np.random.randint(0, len(labeled_anti_expert_states), size=config['batch_size'])
        unlabeled_indices = np.random.randint(0, len(unlabeled_states), size=config['batch_size'])
        expert_indices = np.random.randint(0, len(expert_states), size=config['batch_size'])

        if algorithm == 'safedice':
            # alpha = find_alpha(unlabeled_states[unlabeled_indices], unlabeled_actions[unlabeled_indices])
            
            info_dict = imitator.update(
                    unlabeled_init_states[unlabeled_init_indices],
                    labeled_anti_expert_states[labeled_anti_expert_indices],
                    labeled_anti_expert_actions[labeled_anti_expert_indices],
                    labeled_anti_expert_next_states[labeled_anti_expert_indices],
                    unlabeled_states[unlabeled_indices],
                    unlabeled_actions[unlabeled_indices],
                    unlabeled_next_states[unlabeled_indices], np.array(alpha, dtype=np.float32)
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
            
            average_returns, average_costs, evaluation_timesteps, \
            cost_violation_ratio, strict_cost_violation_ratio, cvar_50, cvar_20, cvar_10, cvar_5 = evaluate_vec_env(vec_env, imitator, deterministic=eval_deterministic, \
                                                                                        cost_violation_threshold=cost_violation_threshold, \
                                                                                        strict_cost_violation_threshold=strict_cost_violation_threshold)

            end = time.time()
            print ('vector env time: ', end-start)
            
            # start = time()
            # average_returns, average_costs, evaluation_timesteps, \
            # cost_violation_ratio, strict_cost_violation_ratio, cvar_50, cvar_20, cvar_10, cvar_5 = evaluate(env, imitator, deterministic=eval_deterministic, \
            #                                                                             cost_violation_threshold=cost_violation_threshold, \
            #                                                                             strict_cost_violation_threshold=strict_cost_violation_threshold)
            # end = time()
            # print ('vector env time: ', end-start)
        
            info_dict.update({'eval': average_returns, 'cost': average_costs, 
                              'cost_violation_ratio': cost_violation_ratio,
                              'strict_cost_violation_ratio': strict_cost_violation_ratio,
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

