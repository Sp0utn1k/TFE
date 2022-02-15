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

	def __init__(self,team,idx=0):

		self.team = team
		self.idx = idx

class Environment:

	def __init__(self,args):

		self.actions = create_actions_set(args.N_aim)
		# self.action_space = self.actions.keys().shape
		self.N_red = args.N_red
		self.N_blue = args.N_blue
		self.N_players = args.N_players
		self.obs_size = args.obs_size
		self.size  = args.size
		self.visibility = args.visibility
		self.R50 = args.R50
		self.alive = {}
		self.positions = {}
		self.aim = {}
		self.red_players = [Player('red',idx=i) for i in range(self.N_red)]
		self.blue_players = [Player('blue',idx=i+self.N_red) for i in range(self.N_blue)]
		self.players = self.red_players+self.blue_players
		self.all_players = self.players
		self.obstacles = args.obstacles
		if args.add_graphics:
			self.graphics = Graphics(args)
			self.show_plot = True
			self.twait = 1000.0/args.fps
		else:
			self.graphics = None
			self.show_plot = False
		self.reset()

	def reset(self):
		self.players = self.all_players
		self.red_players = [player for player in self.players if player.team=="red"]
		self.blue_players = [player for player in self.players if player.team=="blue"]
		self.target_list = {}

		for player in self.players:
			self.alive[player] = False

		for player in self.players:
			self.alive[player] = True
			self.aim[player] = None
			self.positions[player] = [np.random.randint(self.size),np.random.randint(self.size)]

			while (self.positions[player] in self.obstacles or
				  self.positions[player] in [self.positions[p] for p in self.players if self.alive[p] and p != player]):	
				
				self.positions[player] = [np.random.randint(self.size),np.random.randint(self.size)]

		self.current_player = self.players[0]

	def observation(self,p1,p2):
		vis = self.is_visible(self.positions[p1],self.positions[p2])
		if vis:
			friend = 2*(p1.team == p2.team)-1
			x2,y2 = self.positions[p2]
			x1,y1 = self.positions[p1]
			rel_pos = [x2-x1,y2-y1]

			# Normalize :
			rel_pos = np.array(rel_pos).astype(np.float64)
			rel_pos *= 1/self.visibility
			x,y = list(rel_pos)

			obs = [vis,p2.idx,friend,x,y]
			return obs

		return [0,0,0,0,0]

	def observations(self,p1):
		obs = []
		for p2 in self.players:
			if not p1 is p2:
				obsi = self.observation(p1,p2)
				obs += obsi

		assert len(obs) <= self.obs_size, "Too much observations"
		pad = [0 for i in range(self.obs_size-len(obs))]
		obs += pad
		return obs

	def get_player_with_id(self,idx):
		return [p for p in self.all_players if p.idx==idx][0]

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
				visibles += [p2.idx]
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
				if self.aim[player].idx in self.visible_targets_id(player):
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
			# print(f'Player {player.idx} ({player.team}) goes {act}')
		if act == 'shoot':
			is_hit = self.fire(player)
			target = self.aim[player]
			print(f'Player {player.idx} ({player.team}) shots at player {target.idx} ({target.team})')
			if is_hit:
				print("hit!")

		if 'aim' in act:
			target_id = int(act[3:])
			self.aim[player] = self.get_player_with_id(target_id)
			if self.show_plot:
				target = self.aim[player]
				# print(f'Player {player.idx} ({player.team}) aims at player {target.idx} ({target.team})')

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

	def step(self,action,prompt_action=False):
			self.action(self.current_player,A)
			if prompt_action:
				print(f'Agent {self.current_player.idx} takes action "{self.actions[action]}".')

	def get_random_action(self):
		return np.random.choice(list(self.actions.keys()))

	def agent_iter(self):
		for p in self.players:
			self.update_players()
			if self.alive[p]:
				self.current_player = p
				yield p.idx

	# # Deprecated:

	# def ask_action(self,player):
	# 	player.control = player.control.lower()
	# 	if player.control == "random":
	# 		act = np.random.choice(list(self.actions.keys()))
	# 		return act
	# 	elif player.control == "human":
	# 		self.show_fpv(player)
	# 		act = input("Select action (nothing,up,down,left,right,aim,shoot): ")
	# 		while act not in ["nothing","up","left","right","down","aim","shoot"]:
	# 			print(f'{act} is not a valid action.')
	# 			act = input("Select action (nothing,up,down,left,right,aim,shoot): ")
	# 		if act == 'aim':
	# 			target = input("Select target idx you want to aim at: ")
	# 			while not target.isnumeric() or int(target) >= self.N_players:
	# 				print(f'{target} is not a valid idx (only numbers < {self.N_players}).')
	# 				target = input("Select target idx you want to aim at: ")
	# 			act += target
	# 		act = [k for k in self.actions.keys() if self.actions[k] == act][0]
	# 		return act
	# 	else:
	# 		print(f'Player control mode {player.control} does not exist.')

	def update_players(self):
		self.players = [p for p in self.players if self.alive[p]]
		# np.random.shuffle(self.players)
		self.red_players = [p for p in self.players if p.team=="red"]
		self.blue_players = [p for p in self.players if p.team=="blue"]

	def episode_over(self):
		if len(self.blue_players) == 0 or len(self.red_players) == 0:
			return True
		return False

	def render(self,twait=0):
		if not self.show_plot:
			return
		self.graphics.reset()
		for [x,y] in self.obstacles:
			self.graphics.set_obstacle(x,y)
		for player in self.players:
			idx = player.idx
			team = player.team	
			pos = self.positions[player]
			self.graphics.add_player(idx,team,pos)
		cv.imshow('image',self.graphics.image)
		cv.waitKey(round(twait))
		# cv.destroyAllWindows()

	def show_fpv(self,player,twait=0):
		if not self.show_plot:
			return
		self.graphics.reset()
		for [x,y] in self.obstacles:
			self.graphics.set_obstacle(x,y)
		for p in self.players:
			idx = p.idx
			team = p.team
			pos = self.positions[p]
			self.graphics.add_player(idx,team,pos)
		for x in range(self.size):
			for y in range(self.size):
				if not self.is_visible(self.positions[player],[x,y]):
					self.graphics.delete_pixel(x,y)
		# for [x,y] in los(self.positions[player],[35,22]):
		# 	self.graphics.set_red(x,y)

		cv.imshow('image',self.graphics.image)
		cv.waitKey(round(twait))
		# cv.destroyAllWindows()

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

	def add_player(self,idx,team,pos):
		[x,y] = pos
		if team=='red':
			self.set_red(x,y)
		elif team=='blue':
			self.set_blue(x,y)
		cv.putText(self.image,f'{idx}',self.center(x,y),cv.FONT_HERSHEY_SIMPLEX,0.6*self.szi/15,(0,0,0),2)

	def erase_tile(self,x,y):
		self.assign_value(x,y,self.background_color)

def norm(vect1,vect2):
	x1,y1 = vect1
	x2,y2 = vect2
	res = (x2-x1)**2 + (y2-y1)**2
	return math.sqrt(res)

def los(vect1,vect2):
def get_los(vect1,vect2,los_dict):

    if vect2[0] < vect1[0]:
        vect1,vect2 = vect2,vect1

    diff = [vect2[0]-vect1[0],vect2[1]-vect1[1]]
    mirrored = False
    if diff[1] < 0:
        mirrored = True
        diff[1] = -diff[1]

    los = [[i+vect1[0],j*(-1)**mirrored+vect1[1]] for [i,j] in los_dict[tuple(diff)]]
    return los
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
	# 		# env.show_fpv(env.get_player_with_id(0),twait=500)
	# 		env.render(twait=500)

	print(env.observations(p1))
	print(env.actions)
	env.show_fpv(p1)

	# # p1.control = "human"
	# while not env.episode_over():
	# 	env.step()
	# 	# env.show_fpv(p1,twait=1000)
	# 	env.render(twait=1000)