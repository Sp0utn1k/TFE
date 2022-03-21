import yaml
import tanksEnv
from utils.networks.FCNetwork import FCNetwork
from utils.algorithms.DQN import Agent, DQNRunner 
from tensorboardX import SummaryWriter
import time

if __name__ == "__main__":

	# parser = argparse.ArgumentParser(description='Test script for the tanksEnv environment.')
	# parser.add_argument('--MA',metavar='ma',type=bool,default=False,help='Multi Agent mode (default: False)')
	# parser.add_argument('--size',metavar=('size0','size1'),default=[10,10],nargs=2,type=int,
	# 	help='Size of the grid (2D) separated by a space (default: 10 10)')
	# parser.add_argument('--visibility',metavar='vis',default=8,type=float,help='Set Maximum visible distance (default: 8)')
	# parser.add_argument('--R50',metavar='r',default=7,type=float,help='Set distance at which there is a 50%% hit probability (default: 7)')
	# args = vars(parser.parse_args())
	# print(args)

	with open('./configs/dqn_config.yml','r') as file:
		settings = yaml.safe_load(file)

	env = tanksEnv.Environment(**settings)
	S = env.reset()
	obs_size = len(S)
	n_actions = env.n_actions

	agent_net = FCNetwork(obs_size,n_actions,**settings)
	agent = Agent(network=agent_net,**settings)
	runner = DQNRunner(env,agent,**settings)

	timestr = time.strftime('%Y_%m_%d-%Hh%M')
	writer = SummaryWriter('runs/simple DQN/'+timestr)
	
	for (episode,episode_length,reward,loss) in runner.run(settings['n_episodes'],render=settings.get('render',False)):
		writer.add_scalar('reward',reward,episode)
		if loss != None:
			writer.add_scalar('loss',loss,episode)