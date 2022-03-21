import torch
import torch.nn as nn
import random
import numpy as np
from collections import namedtuple, deque
import math
import copy

Episode = namedtuple('Episode',['state','action','reward','next_state','done','all_states','index'])

class Agent:

	def __init__(self,*args,**kwargs):
		
		self.device = torch.device("cuda" if torch.cuda.is_available() and kwargs.get('use_gpu',False) else "cpu")
		print(f'Device: {self.device}')

		self.net = kwargs['network'].to(self.device)
		self.target_net = copy.deepcopy(self.net)
		self.target_net.eval()
		
		self.epsilon = kwargs.get('epsilon',.5)
		self.gamma = kwargs.get('gamma',.9)

		self.loss_fn = kwargs.get('loss_fn',nn.MSELoss())
		self.optimizer = kwargs.get('optimizer',torch.optim.Adam)(self.net.parameters(),lr=kwargs.get('lr',1e-3))
		self.n_actions = None

		if 'epsilon_decay' in kwargs.keys():
			self.eps_min = kwargs['epsilon_decay']['stop']
			self.eps_max = kwargs['epsilon_decay']['start']
			self.eps_period = kwargs['epsilon_decay']['period']
			self.eps_decay_shape = kwargs['epsilon_decay']['shape']
			self.epsilon = self.eps_max
			
		self.use_rnn = kwargs.get('use_rnn',False)

	def sync_nets(self):
		self.target_net.load_state_dict(self.net.state_dict())

	def set_epsilon(self,t):
		shape = self.eps_decay_shape.lower()
		if shape == 'exponential':
			rate = math.log(self.eps_min/self.eps_max)/self.eps_period
			epsilon = self.eps_max*math.exp(t*rate)
		elif shape == 'linear':
			rate = (self.eps_max-self.eps_min)/self.eps_period
			epsilon = self.eps_max - t*rate
		else:
			print('Unknown epsilon decay shape')
		self.epsilon = max(self.eps_min,epsilon)

	def get_action(self,state,hidden):
		if self.n_actions == None:
			self.n_actions = self.net(state.to(self.device)).size(-1)
		assert(state.shape[0] == 1)
		if random.random() < self.epsilon:
			action = random.randrange(self.n_actions)
			next_hidden = hidden
		else:
			with torch.no_grad():
				action,next_hidden = self.net(state.to(self.device)).cpu().squeeze().argmax().numpy()
		return int(action), next_hidden

	def train_net(self,batch):

		device = self.device
		state = torch.cat([episode.state for episode in batch])
		action = torch.tensor([episode.action for episode in batch],device=device)
		reward = torch.tensor([episode.reward for episode in batch],device=device)
		next_state = torch.cat([episode.next_state for episode in batch])
		done = torch.cuda.BoolTensor([episode.done for episode in batch])

		with torch.no_grad():
			Q_ = self.target_net(next_state).max(1)[0]
			Q_[done] = 0.0
			Q_ = Q_.detach()
		
		Q = self.net(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
		target_Q = reward + self.gamma*Q_

		loss = self.loss_fn(Q,target_Q)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()

class Buffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, BATCH_SIZE):
        indices = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return batch

class DQNRunner:
	def __init__(self,env,agent,**kwargs):
		self.env = env
		self.agent = agent
		self.buffer = Buffer(kwargs.get('buffer_size',None))
		self.batch_size = kwargs.get('batch_size',10)
		self.device = self.agent.device
		self.net_sync_period = kwargs.get('net_sync_period',1)

	def run(self,N_episodes,render=False):

		agent = self.agent
		env = self.env
		for episode_id in range(N_episodes):
			total_reward = 0.0
			state = torch.tensor(env.reset(),device=self.device,dtype=torch.float32).unsqueeze(0)
			done = False
			episode_length = 0.0
			loss = 0.0
			all_states = []
			while not done:
				episode_length += 1.0
				agent.set_epsilon(episode_id)
				action = agent.get_action(state)
				next_state, reward, done, _ = env.step(action)
				next_state = torch.tensor(next_state,device=self.device,dtype=torch.float32).unsqueeze(0)
				all_states.append(state)
				episode = Episode(state,action,reward,next_state,done,all_states,len(all_states)-1)
				self.buffer.append(episode)
				state = copy.deepcopy(next_state)
				total_reward += reward
				if loss != None and len(self.buffer) >= self.batch_size:
					loss += agent.train_net(self.buffer.sample(self.batch_size))
				else:
					loss = None
				if render:
					env.render()

			if episode_id % self.net_sync_period == 0:
				agent.sync_nets()

			if loss != None:
				loss /= episode_length

			yield episode_id,episode_length,total_reward,loss