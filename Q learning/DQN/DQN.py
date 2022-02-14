import torch
import torch.nn as nn
import random
import numpy as np
from collections import namedtuple, deque
import gym
import matplotlib
import matplotlib.pyplot as plt
# plt.ion()
import math
import time
import copy


Episode = namedtuple('Episode',['S','A','R','S_','done'])
BATCH_SIZE = 128
BUFFER_SIZE = 5000
N_episodes = 12000
POINTS_ON_PLOT = 500
N_demos = 5

EPSILON_DECAY = {
				'period': 10000,
				'start':1,
				'stop':.02,
				'shape':'exponential'
				}

GAMMA = .999
ALPHA = 1e-3
NET_SYNC_PERIOD = 10
PRINT_PROGRESS_PERIOD = 1000

class NeuralNetwork(nn.Module):
	def __init__(self,n_inputs,n_outputs):
		super(NeuralNetwork,self).__init__()
		self.pipe = nn.Sequential(
			nn.Linear(n_inputs,64),
			nn.ReLU(),
			nn.Linear(64,64),
			nn.ReLU(),
			nn.Linear(64,n_outputs)
		)
	def forward(self,x):
		return self.pipe(x)

class Agent:

	def __init__(self,*args,**kwargs):
		assert('n_states' and 'n_actions' in kwargs.keys())
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f'Device: {self.device}')
		self.n_states = kwargs['n_states']
		self.n_actions = kwargs['n_actions']
		self.net = NeuralNetwork(self.n_states,self.n_actions).to(self.device)
		self.net2 = NeuralNetwork(self.n_states,self.n_actions).to(self.device)
		self.net2.eval()
		
		self.epsilon = kwargs.get('epsilon',.5)
		self.gamma = kwargs.get('gamma',.9)
		self.alpha = kwargs.get('alpha',1e-3)

		self.loss_fn = nn.MSELoss()
		# self.loss_fn = nn.SmoothL1Loss()
		self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.alpha)

		if 'epsilon_decay' in kwargs.keys():
			self.eps_min = kwargs['epsilon_decay']['stop']
			self.eps_max = kwargs['epsilon_decay']['start']
			self.eps_period = kwargs['epsilon_decay']['period']
			self.eps_decay_shape = kwargs['epsilon_decay']['shape']
			self.epsilon = self.eps_max

	def sync_nets(self):
		self.net2.load_state_dict(self.net.state_dict())

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

	def get_action(self,S):
		assert(S.shape[0] == 1)
		if random.random() < self.epsilon:
			A = random.randrange(self.n_actions)
		else:
			with torch.no_grad():
				A = self.net(S.to(self.device)).cpu().squeeze().argmax().numpy()
		return int(A)

	def train_net(self,batch):

		device = self.device
		S = torch.cat([episode.S for episode in batch])
		A = torch.tensor([episode.A for episode in batch],device=device)
		# print(sum(A)/len(A))
		R = torch.tensor([episode.R for episode in batch],device=device)
		S_ = torch.cat([episode.S_ for episode in batch])
		done = torch.cuda.BoolTensor([episode.done for episode in batch])

		# print(f'S {S.size()}, A {A.size()}, R {R.size()}, S_ {S_.size()}, done {done.size()}')
		with torch.no_grad():
			Q_ = self.net2(S_).max(1)[0]
			Q_[done] = 0.0
			Q_ = Q_.detach()
		
		Q = self.net(S).gather(1, A.unsqueeze(-1)).squeeze(-1)
		target_Q = R + self.gamma*Q_

		loss = self.loss_fn(Q,target_Q)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

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

def append(batch,episode):
	assert(isinstance(episode,Episode))
	batch = np.append(batch,np.array([episode],dtype=Episode))
	return batch

if __name__ == "__main__":

	env = gym.make("CartPole-v0")
	obs_space = env.observation_space.shape[0]
	n_actions = env.action_space.n

	agent = Agent(n_states=obs_space,n_actions=n_actions,gamma=GAMMA,alpha=ALPHA,epsilon_decay=EPSILON_DECAY)
	buffer = Buffer(BUFFER_SIZE)

	total_R = 0
	R_plot = []
	S = torch.tensor(env.reset(),device=agent.device).unsqueeze(0)
	for step in range(N_episodes):
		# env.render()
		agent.set_epsilon(step)
		A = agent.get_action(S)
		S_, R, done, _ = env.step(A)
		S_ = torch.tensor(S_,device=agent.device).unsqueeze(0)
		episode = Episode(S,A,R,S_,done)
		buffer.append(episode)

		S = copy.deepcopy(S_)
		total_R += R

		if done:
			R_plot += [total_R]
			total_R = 0
			S = torch.tensor(env.reset(),device=agent.device).unsqueeze(0)

		if step % PRINT_PROGRESS_PERIOD == 0:
			print(f'step {step} / {N_episodes}')
			# for param in agent.net.parameters():
  	# 			print(param)


		if len(buffer) < BATCH_SIZE:
			continue

		agent.train_net(buffer.sample(BATCH_SIZE))

		if step % NET_SYNC_PERIOD == 0:
			agent.sync_nets()
		
	if len(R_plot) > POINTS_ON_PLOT:
		K = int(len(R_plot)//POINTS_ON_PLOT)
		R_reduced = R_plot[0::K]
		t = [K*i for i in range(len(R_reduced))]
		plt.plot(t,R_reduced)
	else:
		plt.plot(R_plot)
		pass

	plt.show()

	# Demo:
	agent.epsilon = 0
	for _ in range(N_demos):
		done = False
		S = torch.tensor(env.reset(),device=agent.device).unsqueeze(0)
		while not done:
			env.render()
			A = agent.get_action(S)
			S, _, done, _ = env.step(A)
			S = torch.tensor(S,device=agent.device).unsqueeze(0)
