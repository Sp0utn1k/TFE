class Args:
	
	# Environment
	size = 11
	observation_map_size = 25

	players_description = [
		{'pos0':'random','team':'red','replicas':2},
		{'pos0':'random','team':'blue','replicas':2}
	]

	# players_description = [
	# 	{'pos0':[0,0],'team':'red'},
	# 	{'pos0':[0,9],'team':'red'},
	# 	{'pos0':[9,0],'team':'blue'},
	# 	{'pos0':[9,9],'team':'blue'}
	# ]

	visibility = 20
	R50 = .8*visibility
	obstacles = [[x,y] for x in range(1,10) for y in [5]]
	# obstacles = []

	# Agent
	N_aim = 4
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