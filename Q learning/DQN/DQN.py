import torch
import torch.nn as nn
import random
import numpy as np
from collections import namedtuple, deque
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import time
import copy

Episode = namedtuple('Episode',['S','A','R','S_','done'])
batch_size = 200
BUFFER_SIZE
EPOCHS = 2000
POINTS_ON_PLOT = 500

EPSILON_DECAY = {
				'period':.9*EPOCHS,
				'start':1,
				'stop':.02
				}
GAMMA = .99
ALPHA = 1e-3
NET_SYNC_PERIOD = 20

class NeuralNetwork(nn.Module):
	def __init__(self,n_inputs,n_outputs):
		super(NeuralNetwork,self).__init__()
		self.pipe = nn.Sequential(
			nn.Linear(n_inputs,64),
			nn.ReLU(),
			nn.Linear(64,32),
			nn.ReLU(),
			nn.Linear(32,n_outputs),
			nn.Softmax(dim=1)
		)
	def forward(self,x):
		return self.pipe(x)

class Agent:

	def __init__(self,*args,**kwargs):
		assert('n_states' and 'n_actions' in kwargs.keys())
		
		self.n_states = kwargs['n_states']
		self.n_actions = kwargs['n_actions']
		self.net = NeuralNetwork(self.n_states,self.n_actions)
		self.net2 = NeuralNetwork(self.n_states,self.n_actions)
		
		self.epsilon = kwargs.get('epsilon',.5)
		self.gamma = kwargs.get('gamma',.9)
		self.alpha = kwargs.get('alpha',1e-3)

		self.loss_fn = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.alpha)

		if 'epsilon_decay' in kwargs.keys():
			self.eps_min = kwargs['epsilon_decay']['stop']
			self.eps_max = kwargs['epsilon_decay']['start']
			self.eps_rate = math.log(self.eps_min/self.eps_max)/kwargs['epsilon_decay']['period']


	def sync_nets(self):
		self.net2.load_state_dict(self.net.state_dict())

	def set_epsilon(self,t):
		self.epsilon = max(self.eps_min,self.eps_max*math.exp(t*self.eps_rate))

	def get_action(self,S):
		assert(S.shape[0] == 1)
		if random.random() < self.epsilon:
			A = random.randrange(self.n_actions)
		else:
			A = self.net(S).squeeze().argmax().numpy()
		return int(A)

	def update(self,batch,device='cpu'):
		
		S = torch.cat([episode.S for episode in batch]).to(device)
		A_mask = [list(np.arange(len(batch))),[episode.A for episode in batch]]
		# A_mask = torch.cat((A,torch.arange(len(A)))).reshape(2,-1).transpose(0,1)
		A = torch.zeros((len(batch),self.n_actions),dtype=torch.bool).to(device)
		A[A_mask] = True
		R = torch.tensor([episode.R for episode in batch]).to(device)
		S_ = torch.cat([episode.S_ for episode in batch]).to(device)
		done = torch.tensor([episode.done for episode in batch]).to(device)

		with torch.no_grad():
			Q_ = self.net2(S_)
			Q_ = ~done*torch.max(Q_,dim=1).values.detach()
		
		Q = self.net(S)[A]
		target = R + Q_*self.gamma

		loss = self.loss_fn(Q,target)
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

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]


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
	for epoch in range(EPOCHS):
		batch = []
		done = False
		S = torch.tensor(env.reset()).unsqueeze(0)
		while not done:
			A = agent.get_action(S)
			S_, R, done, _ = env.step(A)
			total_R += R
			S_ = torch.tensor(S_).unsqueeze(0)
			S = S_
			episode = Episode(S,A,R,S_,done)
			buffer.append(episode)

		R_plot += [total_R]
		total_R = 0
		if epoch % NET_SYNC_PERIOD == 0:
			agent.sync_nets()
			print(f'{epoch}/{EPOCHS}')

		agent.update(batch,device='cpu')

	if len(R_plot) > POINTS_ON_PLOT:
		K = int(len(R_plot)//POINTS_ON_PLOT)
		R_reduced = R_plot[0::K]
		t = [K*i for i in range(len(R_reduced))]
		plt.plot(t,R_reduced)
	else:
		plt.plot(R_plot)
	plt.savefig("first_pytorch.png")