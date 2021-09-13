import torch
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm import tqdm
import pickle
from math import floor

class DemonstrationData(torch.utils.data.Dataset):
	def __init__(self, data_dir, trajectories, pov_shape, vec_shape, action_shape):
		self.path = data_dir
		self.collate_data(trajectories)

		# info about data shapes
		self.pov_shape = pov_shape
		self.vec_shape = vec_shape
		self.action_shape = action_shape

	def __len__(self):
		return self.N_ -1
	
	def __getitem__(self, index):
		obfvec = self.states[index]
		pov = self.load_frame(index)
		action = self.actions[index]
		
		return {'pov': pov.float(), 'vector': torch.from_numpy(obfvec)}, {'action': torch.from_numpy(action)}

	def load_frame(self, idx):
		pov = torch.load(self.path+'/povs/id-%d.pt' % idx)
		return pov

	def collate_data(self, trajectories):
		self.states = trajectories[0]['vector']
		self.actions = trajectories[0]['action']
		self.idx = trajectories[0]['id']
		self.episode_len = np.arange(trajectories[0]['id'].shape[0])

		print('Collating demonstration data for dataset...')
		for i in tqdm(range(1, len(trajectories))):
			self.states = np.vstack((self.states, trajectories[i]['vector']))
			self.actions = np.vstack((self.actions, trajectories[i]['action']))
			self.idx = np.hstack((self.idx, trajectories[i]['id']))
			self.episode_len = np.hstack((self.episode_len, np.arange(trajectories[i]['id'].shape[0])))

		self.N_ = self.states.shape[0]

	# def transforms(self):
	# 	transform = transforms.Compose([
	# 		transforms.Normalize()
	# 	])

	#	return transform
		
def train_val_split(data_dir, holdout=0.2, trim=True):
	with open(data_dir+'/demonstrations.pkl', 'rb') as f:
		demonstration_data = pickle.load(f)
	states = demonstration_data['vector'].astype(np.float32)
	actions = demonstration_data['action'].astype(np.float32)
	dones = demonstration_data['done']
	
	trajectories = []
	n_samples = states.shape[0]
	traj_start_idx = 0
	traj_end_idx = 0
	for i in range(n_samples):
		traj_end_idx += 1
		if dones[i]:
			traj_data = {'vector': states[traj_start_idx:traj_end_idx],
							'action': actions[traj_start_idx:traj_end_idx],
							'id': np.arange(start=traj_start_idx, stop=traj_end_idx)}
			trajectories.append(traj_data)

			traj_start_idx = traj_end_idx          

	if holdout > 0.0:
		idx = np.arange(len(trajectories), dtype=np.int)
		idx = np.random.permutation(idx)
		test_idx, train_idx = idx[0:floor(holdout*len(idx))], idx[floor(holdout*len(idx)):]

		train_trajectories = [trajectories[i] for i in train_idx]
		test_trajectories = [trajectories[i] for i in test_idx]

		return train_trajectories, test_trajectories
	else:
		return trajectories

def process_trajectories(envs, path, trim=True, trim_reward=11):
	# make directory to store trajectories
	try:
		os.mkdir(path)
	except FileExistsError:
		# os.remove(path+'/demonstrations.pkl')
		pass

	data = {'vector': [], 'action': [], 
			'reward': [], 'next_vector': [], 'done': []}
	pov_data = []
	next_pov_data = []

	import minerl
	for env in envs:
		env_data = minerl.data.make(env)
		trajectories = env_data.get_trajectory_names()
		for traj in trajectories:
			try:
				traj_reward = 0
				# sample trajectory only if its un-corrupted
				for i, sample in enumerate(env_data.load_data(traj, include_metadata=True)):
					if i == 0:
						meta_data = sample[5]
						# if trimming trajectories, skip those that dont meet required reward
						if meta_data['total_reward'] < trim_reward:
							break

					pov_data.append(sample[0]['pov'])
					data['vector'].append(sample[0]['vector'])
					data['action'].append(sample[1]['vector'])
					data['reward'].append(sample[2])
					next_pov_data.append(sample[3]['pov'])
					data['next_vector'].append(sample[3]['vector'])
					data['done'].append(sample[4])

					traj_reward += sample[2]

					# if trimming break when required reward is met
					if trim and traj_reward >= trim_reward:
						# makr the end of trimmed trajectory
						data['done'][-1] = True
						break

			except TypeError:
				# sometimes trajectory file is corrupted, if so skip it
				pass
	
	# convert lists into numpy arrays
	for key in data.keys():
		data[key] = np.array(data[key])
	
	# save trajectory data
	filename = path+'/demonstrations.pkl'
	with open(filename, 'wb') as f:
		pickle.dump(data, f)
	
	assert len(pov_data) == len(next_pov_data), 'check data processing, lens of `pov_data` and `next_pov_data` not same.'
	# save pov data
	pov_dir = path+'/povs'
	next_pov_dir = path+'/next_povs'
	print("Saving frames to {}, {}...".format(pov_dir, next_pov_dir))
	
	try:
		os.mkdir(pov_dir)
	except FileExistsError:
		for f in os.listdir(pov_dir):
			os.remove(pov_dir+'/'+f)
	try:
		os.mkdir(next_pov_dir)
	except FileExistsError:
		for f in os.listdir(next_pov_dir):
			os.remove(next_pov_dir+'/'+f)
	
	for i in tqdm(range(len(pov_data))):
		pov = torch.from_numpy(pov_data[i])
		next_pov = torch.from_numpy(next_pov_data[i])
		filename = pov_dir+'/id-{}.pt'.format(i)
		torch.save(pov, filename)
		filename = next_pov_dir+'/id-{}.pt'.format(i)
		torch.save(next_pov, filename)

# convert dict(tuple/list) of numpy arrays to dict(tuple) of torch tensors
def convert_to_tensors(*args, device):
	if len(args) == 1:
		# dict was passed, return dict
		if isinstance(args[0], dict):
			tensors = {}
			for key, value in args[0].items():
				tensors[key] = convert_to_tensors(value, device=device)

		# tuple or list was passed, return tuple
		elif isinstance(args[0], tuple) or isinstance(args[0], list):
			tensors = tuple([convert_to_tensors(arr, device=device) for arr in args[0]])

		# single array was passed
		elif isinstance(args[0], np.ndarray):
			tensors = torch.from_numpy(args[0])

		elif isinstance(args[0], torch.Tensor):
			tensors = args[0]

		else:
			raise TypeError('{} object cannot be converted to tensor'.format(type(args[0])))
		
	else:
		tensors = tuple([convert_to_tensors(arg, device=device) for arg in args])
	
	tensors = buffer_to(tensors, device)
	return tensors

def convert_to_numpy(*args):
	if len(args) == 1:
		if isinstance(args[0], list) or isinstance(args[0], tuple):
			arrays = tuple([convert_to_numpy(arr) for arr in args[0]])
	
		elif isinstance(args[0], dict):
			arrays = {}
			for key, val in args[0].items():
				arrays[key] = convert_to_numpy(val)
		
		elif isinstance(args[0], torch.Tensor):
			arrays = args[0].cpu().numpy()
		
		elif isinstance(args[0], np.ndarray):
			arrays = args[0]
		
		else:
			raise TypeError('unsupported type {} for conversion to np.ndarray'.format(type(args[0])))
	
	else:
		arrays = tuple([convert_to_numpy(arg) for arg in args])
	
	return arrays

def minerl_inputs_format(state):
	pov = state['pov']
	# rescale
	pov = (1/255)*pov.astype(np.float32)

	state['pov'] = pov
	return state


def buffer_to(buffer, device):
	if isinstance(buffer, torch.Tensor):
		return buffer.to(device)
	elif isinstance(buffer, list) or isinstance(buffer, tuple):
		return tuple([buffer_to(b, device) for b in buffer])
	elif isinstance(buffer, dict):
		for key, value in buffer.items():
			buffer[key] = buffer_to(value, device)
		return buffer
	else:
		raise TypeError('cant move {} to device {}'.format(type(buffer), device))

def obs_transforms(obs):
	scale = 1 / 255
	pov, vector = obs['pov'].astype(np.float32), obs['vector'].astype(np.float32)
	
	num_elem = pov.shape[-3] * pov.shape[-2]
	vector_channel = np.tile(vector, num_elem // vector.shape[-1]).reshape(*pov.shape[:-1], -1)  # noqa
	return np.concatenate([pov / scale, vector_channel / scale], axis=-1)
	
if __name__ == "__main__":
	# data_dir = 'D:/IIT/mineRL/data/bc/Navigate'
	# train, test = train_val_split(data_dir)
	# train = DemonstrationData(data_dir, train)

	# print(train.N_)
	# print(train.idx.shape)
	process_trajectories(envs='ObtainDiamond', path="/media/user/997211ec-8c91-4258-b58e-f144225899f4/MinerlV2/dhruvlaad/data/ObtainDiamond", trim = True)
	train_val_split(data_dir="/media/user/997211ec-8c91-4258-b58e-f144225899f4/MinerlV2/dhruvlaad/data/ObtainDiamond", holdout=0.2, trim=True)
