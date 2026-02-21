import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'glfw'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
from plot_metrics import plot_metrics,plot_K,save_K

import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	return np.nanmean(episode_rewards)


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
 
        
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
	
	# Run training
	L = logger.Logger(work_dir, cfg)
	episode_idx, start_time = 0, time.time()
	grad_cov_rank = 0
	grad_cov_frob = 0
	eNTK_rank = 0
	eNTK_frob = 0
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			episode += (obs, action, reward, done)
		assert len(episode) == cfg.episode_length
		buffer += episode

		#compute_K = True
		compute_K = (episode_idx%20==0)

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates-1):
				agent.update(buffer, step+i)
			train_metrics.update(agent.update(buffer, step+i,compute_metrics=True,compute_K=compute_K))
			print('End update')
			if compute_K:
				eNTK = train_metrics['eNTK']
				K = train_metrics['K']
				path_save_K = save_K(K,work_dir,env_step,name='K')
				path_save_eNTK = save_K(eNTK,work_dir,env_step,name='eNTK')
				plot_K(path_save_K,work_dir,env_step,name="K")
				plot_K(path_save_eNTK,work_dir,env_step,name='eNTK')
				grad_cov_rank = torch.linalg.matrix_rank(K.cpu(),hermitian=True).item()
				eNTK_rank = torch.linalg.matrix_rank(eNTK.cpu(),hermitian=True).item()
				grad_cov_frob = torch.norm(K, 'fro').item()
				eNTK_frob = torch.norm(eNTK, 'fro').item()
			train_metrics['grad_cov_rank'] = grad_cov_rank
			train_metrics['grad_cov_frob'] = grad_cov_frob
			train_metrics['eNTK_rank'] = eNTK_rank
			train_metrics['eNTK_frob'] = eNTK_frob
   
		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)

		L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
			L.log(common_metrics, category='eval')
			plot_metrics([str(work_dir)],name="exp",labels=['exp']) 

	L.finish(agent)
	print('Training completed successfully')


if __name__ == '__main__':
	train(parse_cfg(Path().cwd() / __CONFIG__))
