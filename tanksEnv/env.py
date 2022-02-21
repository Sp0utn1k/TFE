import numpy as np
import math,time,sys,os
from collections import namedtuple
import copy
import pickle

import torch
from torch import nn, optim
import cv2 as cv


with open('los100.pkl', 'rb') as file:
    los_dict = pickle.load(file)

class Player:

	def __init__(self,team,id=0):

		self.team = team
		self.id = id

class Environment:

	def __init__(self,args):

		self.actions = create_actions_set(args.N_aim)
		# self.action_space = self.actions.keys().shape
		self.players_description = args.players_description
		self.size  = args.size
		self.visibility = args.visibility
		self.R50 = args.R50
		self.observation_map_size = args.observation_map_size

		# self.red_players = [Player('red',id=i) for i in range(self.N_red)]
		# self.blue_players = [Player('blue',id=i+self.N_red) for i in range(self.N_blue)]
		# self.players = self.red_players+self.blue_players
		# self.all_players = self.players

		self.obstacles = args.obstacles
		if args.add_graphics:
			self.graphics = Graphics(args)
			self.show_plot = True
			self.twait = 1000.0/args.fps
		else:
			self.graphics = None
			self.show_plot = False
		self.reset()
		self.obs_size = len(self.observe_state(self.players[0],pad=False))

	def reset(self):
		self.init_players()
		self.red_players = [player for player in self.players if player.team=="red"]
		self.blue_players = [player for player in self.players if player.team=="blue"]
		self.alive = {}
		self.aim = {}

		for player in self.players:
			self.alive[player] = True
			self.aim[player] = None
			if not player in self.positions.keys():
				self.positions[player] = [np.random.randint(self.size),np.random.randint(self.size)]

				while (self.positions[player] in self.obstacles or
					  self.positions[player] in [self.positions[p] for p in self.players 
					  if p in self.positions.keys() and self.alive[p] and p != player]):	
					
					self.positions[player] = [np.random.randint(self.size),np.random.randint(self.size)]

		# np.random.shuffle(self.players)
		self.current_player = self.players[0]

	def init_players(self):
		self.positions = {}
		players_description = self.players_description
		self.players = []
		for d in players_description:
			team = d['team']
			pos0 = d.get('pos0','random')
			replicas = d.get('replicas',1)
			assert team in ['blue','red'], 'Team must be red or blue'
			for _ in range(replicas):
				player = Player(team,id=len(self.players))
				self.players += [player]
			if not pos0 == 'random':
				assert replicas == 1, 'Cannot replicate player with fixed position'
				self.positions[player] = pos0

	def observe_other(self,player,other):
		vis = self.is_visible(self.positions[player],self.positions[other])
		if vis:
			friend = (player.team == other.team)
			x2,y2 = self.positions[other]
			x1,y1 = self.positions[player]
			rel_pos = [x2-x1,y2-y1]

			# Normalize :
			rel_pos = np.array(rel_pos).astype(np.float64)
			rel_pos *= 1.0/self.visibility
			x,y = list(rel_pos)

			obs = [vis,friend,x,y]
			return obs

		return [0,0,0,0]

	def observe_state(self,player,pad=True):

		obs = []
		# Self:

		obs = copy.copy(self.positions[player])
		obs += list(np.array(self.obstacles_array).flatten())

		for other in self.players:
			if not player is other:
				obsi = self.observe_other(player,other)
				obs += obsi

		if pad:
			assert len(obs) <= self.obs_size, "Too much observations"
			pad = [0 for i in range(self.obs_size-len(obs))]
			obs += pad
		return obs

	def get_player_with_id(self,id):
		return [p for p in self.players if p.id==id][0]

	def is_visible(self,pos1,pos2):
		if pos1 == pos2:
			return True
		dist = norm(pos1,pos2)
		if dist >= self.visibility:
			return False
		LOS = los(pos1,pos2)
		for pos in LOS:
			if pos in self.obstacles:
				return False
		return True

	def visible_targets_id(self,p1):
		visibles = []
		for p2 in self.players:
			if p2!=p1 and self.is_visible(self.positions[p1],self.positions[p2]):
				visibles += [p2.id]
		return visibles

	def next_tile(self,player,act):
		tile = self.positions[player]
		x,y = tile
		assert act in ['up','down','left','right'], "Unauthorized movement"
		if act =='up':
			y -= 1
		elif act == 'down':
			y += 1
		elif act == 'left':
			x -= 1
		else:
			x += 1
		return [x,y]

	def is_free(self,tile):
		x,y = tile
		if x < 0 or x >= self.size or y < 0 or y >= self.size:
			return False
		if [x,y] in self.obstacles:
			return False
		if [x,y] in self.positions.values():
			return False
		return True

	def is_valid_action(self,player,act):
		act = self.actions[act]
		if act == 'nothing':
			return True
		if act in ['up','down','left','right']:
			return self.is_free(self.next_tile(player,act))
		if act == 'shoot':
			if self.aim[player] in self.players:
				if self.aim[player].id in self.visible_targets_id(player):
					return True
		if 'aim' in act:
			target = int(act[3:])
			if target in self.visible_targets_id(player):
				return True
		return False

	def action(self,player,action):
		if not self.is_valid_action(player,action):
			return
		act = self.actions[action]
		if act in ['up','down','left','right']:
			self.positions[player] = self.next_tile(player,act)
			# print(f'Player {player.id} ({player.team}) goes {act}')
		if act == 'shoot':
			is_hit = self.fire(player)
			target = self.aim[player]
			print(f'Player {player.id} ({player.team}) shots at player {target.id} ({target.team})')
			if is_hit:
				print("hit!")

		if 'aim' in act:
			target_id = int(act[3:])
			self.aim[player] = self.get_player_with_id(target_id)
			if self.show_plot:
				target = self.aim[player]
				print(f'Player {player.id} ({player.team}) aims at player {target.id} ({target.team})')

	def fire(self,player):
		target = self.aim[player]
		distance = norm(self.positions[player],self.positions[target])
		hit = np.random.rand() < self.Phit(distance)
		if hit:
			self.alive[target] = False
			return True
		return False
	
	def Phit(self,r):
		return sigmoid(self.R50-r,12/self.R50)

	def get_reward(self,p):
		winner = self.winner()
		if winner == p.team:
			R = 1
		elif winner == None:
			R = 0
		else:
			R = -1
		return R

	def last(self):
		S = self.observe_state(self.current_player)
		R = self.get_reward(self.current_player)
		done = self.episode_over()
		info = None
		return S_,R,done,info

	def step(self,action,prompt_action=False):
			self.action(self.current_player,A)
			if prompt_action:
				print(f'Agent {self.current_player.id} takes action "{self.actions[action]}".')

	def get_random_action(self):
		return np.random.choice(list(self.actions.keys()))

	def agent_iter(self):
		for p in self.players:
			self.update_players()
			if self.alive[p]:
				self.current_player = p
				yield p.id

	def update_players(self):
		self.players = [p for p in self.players if self.alive[p]]
		# np.random.shuffle(self.players)
		self.red_players = [p for p in self.players if p.team=="red"]
		self.blue_players = [p for p in self.players if p.team=="blue"]

	def winner(self):
		if len(self.blue_players) == 0:
			winner = 'red'
		elif len(self.red_players) == 0:
			winner = 'blue'
		else:
			winner = None
		return winner

	def episode_over(self):
		return self.winner() != None

	def render(self,twait=0):
		if not self.show_plot:
			return
		self.graphics.reset()
		for [x,y] in self.obstacles:
			self.graphics.set_obstacle(x,y)
		for player in self.players:
			id = player.id
			team = player.team	
			pos = self.positions[player]
			self.graphics.add_player(id,team,pos)
		cv.imshow('image',self.graphics.image)
		cv.waitKey(round(twait))
		# cv.destroyAllWindows()

	def show_fpv(self,player_id,twait=0):
		player = self.get_player_with_id(player_id)
		if not self.show_plot:
			return
		self.graphics.reset()
		for [x,y] in self.obstacles:
			self.graphics.set_obstacle(x,y)
		for p in self.players:
			id = p.id
			team = p.team
			pos = self.positions[p]
			self.graphics.add_player(id,team,pos)
		for x in range(self.size):
			for y in range(self.size):
				if not self.is_visible(self.positions[player],[x,y]):
					self.graphics.delete_pixel(x,y)
		# for [x,y] in los(self.positions[player],[35,22]):
		# 	self.graphics.set_red(x,y)

		cv.imshow('image',self.graphics.image)
		cv.waitKey(round(twait))
		# cv.destroyAllWindows()

	@property
	def obstacles_array(self):
		out_size = max(self.observation_map_size,self.size)
		res = np.zeros((out_size,out_size))
		for obstacle in self.obstacles:
			res[tuple(obstacle[::-1])] = 1
		res[self.size,:] = 1
		res[:,self.size] = 1
		return res

class Graphics:
	def __init__(self,args):
		self.size = args.im_size
		self.szi = self.size/args.size
		self.background_color = self.to_color(args.background_color)
		self.red = self.to_color(args.red)
		self.blue = self.to_color(args.blue)
		self.obstacles_color = self.to_color(args.obstacles_color)
		self.reset()

	def reset(self):
		self.image = np.full((self.size,self.size,3),self.background_color,dtype=np.uint8)

	def to_color(self,color):
		(r,g,b) = color
		color = (b,g,r)
		return np.array(color,dtype=np.uint8)

	def pixels_in_coord(self,x,y):
		res = []
		szi = self.szi
		for j in range(round(szi*(x)),round(szi*(x+1))):
			for i in range(round(szi*(y)),round(szi*(y+1))):
					yield [i,j]

	def assign_value(self,x,y,val):
		for c in self.pixels_in_coord(x,y):
			self.image[c[0],c[1],:] = val

	def center(self,x,y):
		return (round(x*self.szi),round((y+.92)*self.szi))

	def set_blue(self,x,y):
		self.assign_value(x,y,self.blue)

	def set_red(self,x,y):
		self.assign_value(x,y,self.red)

	def set_obstacle(self,x,y):
		self.assign_value(x,y,self.obstacles_color)

	def delete_pixel(self,x,y):
		self.assign_value(x,y,[0,0,0])

	def add_player(self,id,team,pos):
		[x,y] = pos
		if team=='red':
			self.set_red(x,y)
		elif team=='blue':
			self.set_blue(x,y)
		cv.putText(self.image,f'{id}',self.center(x,y),cv.FONT_HERSHEY_SIMPLEX,0.6*self.szi/15,(0,0,0),2)

	def erase_tile(self,x,y):
		self.assign_value(x,y,self.background_color)

def norm(vect1,vect2):
	x1,y1 = vect1
	x2,y2 = vect2
	res = (x2-x1)**2 + (y2-y1)**2
	return math.sqrt(res)

def los(vect1,vect2):

    if vect2[0] < vect1[0]:
        vect1,vect2 = vect2,vect1

    diff = [vect2[0]-vect1[0],vect2[1]-vect1[1]]
    mirrored = False
    if diff[1] < 0:
        mirrored = True
        diff[1] = -diff[1]

    los = [[i+vect1[0],j*(-1)**mirrored+vect1[1]] for [i,j] in los_dict[tuple(diff)]]
    return los

def sigmoid(x,l):
	return 1.0/ (1+math.exp(-x*l))

def create_actions_set(N_aim):
	actions = {1:'up',2:'down',3:'left',4:'right'}
	actions[0] = 'nothing'
	actions[5] = 'shoot'
	for i in range(N_aim):
		actions[6+i] = f'aim{i}'
	return actions

if __name__ == "__main__":

	from setup import args
	env = Environment(args)
	# env.update_players()
	p1 = env.get_player_with_id(1)

	# while not env.episode_over():
	# 	for idx in env.agent_iter():
	# 		A = env.get_random_action()
	# 		env.step(A,prompt_action=True)
	# 		# env.show_fpv(0,twait=500)
	# 		env.render(twait=500)

	# print(env.observe_state(p1))
	print(env.obstacles_array)
	# env.show_fpv(1)
	env.render()
	print(env.last())