class Args:
	
	# Environment
	size = 10
	N_blue = 2
	N_red = 2
	N_players = N_red+N_blue
	visibility = 9.0
	R50 = .8*visibility
	obstacles = [[x,y] for x in range(4,6) for y in range(5,6)]

	# Agent
	N_aim = 4
	obs_size = N_aim*5
	dropout = 0.4
	hidden = [256,128,64]

	# Graphics
	add_graphics = True
	im_size = 720
	background_color = [133,97,35]
	red = [255,0,0]
	blue = [30,144,255]
	obstacles_color = [88,86,84]
	fps = 5

args = Args()