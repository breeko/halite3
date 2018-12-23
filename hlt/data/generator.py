import os
import numpy as np
from hlt.data.parse import load_game
from hlt.data.utils import one_hot, stack_relative_frame

def get_input_generator(
	replay_folder: str, 
	player_name: str,
	radius: int,
	look_back: int=0,
	prob_include_frame:float=0.2,
	prob_include_ship:float=0.2,
	batch_size:int=128,
	start_frame_perc:float=0.0,
	end_frame_perc:float=1.0) -> (dict, np.array):
	""""
		Returns an input generator for training a neural network
		inputs:
			replay_folder (str): 						directory that stores .hlt games
			player_name (str): 							name of the player you want to create a training set for
			radius (int): 								how many squares to consider in each direction
			look_back (int): 							how many frames to consider
			prob_include_frame (float - default 0.2): 	probability to include a frame when aggregating frames 
			prob_include_ship (float - default 0.2):  	probability to include a ship when aggregating ships from frames 
			batch_size (int - default 128): size of   	the training batch
			start_frame_perc (float - default 0.0):   	the frame percent to start on (e.g. 0.2 means start 20% through the game)	
			end_frame_perc (float - default 1.0):     	the frame percent to start on (e.g. 0.9 means end after 90%of game is through)

		outputs:
			[{"cargos": cargos, "moves": moves, "halites": halites, "ships": ships, "dropoffs": dropoffs}, outs]

			cargo:    (None, look_back + 1),                                 percent of max capacity a ship is carrying
			turn:     (None, look_back + 1),                                 percent of max turns
			moves:    (None, look_back),		                             moves prior to the latest move
			halite:   (None, (2 * radius + 1), (2 * radius + 1), look_back + 1), halite as percentage of max halite on map
			ships:    (None, (2 * radius + 1), (2 * radius + 1), look_back + 1), ships on map, 1 representing friendly ship and -1 representing enemy ship
			dropoffs: (None, (2 * radius + 1), (2 * radius + 1), look_back + 1), dropoffs on map, 1 representing friendly dropoff and -1 representing enemy dropoff
			outs:     (None, 1),			   			                     still, north, east, south, west as a one-hot vector
			
	"""
	# TODO: Make player_name into a lambda to allow for things like winning player
	def gen():
		map_shape = radius * 2 + 1, radius * 2 + 1
		num_classes = 5

		out_cargos   = np.zeros(shape=(batch_size, look_back + 1, 1), dtype=np.float)
		out_moves    = np.zeros(shape=(batch_size, look_back + 1, num_classes), dtype=str)
		out_halites  = np.zeros(shape=(batch_size, *map_shape, look_back + 1), dtype=np.float)
		out_ships    = np.zeros(shape=(batch_size, *map_shape, look_back + 1), dtype=np.float)
		out_dropoffs = np.zeros(shape=(batch_size, *map_shape, look_back + 1), dtype=np.float)

		ct = 0

		available_files = os.listdir(replay_folder)
		while True:
			file_name = np.random.choice(available_files, size=1)[0]
			file_path = "{}/{}".format(replay_folder, file_name)
			game = load_game(path=file_path,player_name=player_name)
			num_frames = len(game["moves"])

			for num_frame in range(num_frames):
				perc_frame = num_frame / float(num_frames)
				if perc_frame < start_frame_perc:
					continue
				elif perc_frame > end_frame_perc:
					break
				
				if np.random.random() > prob_include_frame:
					continue
				
				moves = game["moves"][num_frame]
				for ship_id in moves.keys():
					if np.random.random() > prob_include_ship:
						continue

					relative_frame = stack_relative_frame(game=game, ship_id=ship_id, frame=num_frame, look_back=look_back, radius=radius)
					out_halites[ct]  = relative_frame["halites"]
					out_ships[ct]    = relative_frame["ships"]
					out_dropoffs[ct] = relative_frame["dropoffs"]
					out_moves[ct]    = one_hot(relative_frame["moves"], num_classes)
					out_cargos[ct]   = np.expand_dims(relative_frame["cargos"],-1)

					ct = (ct + 1) % batch_size
					
					if ct == 0:
						yield {"cargos": np.copy(out_cargos), "moves": np.copy(out_moves[:,:-1, :]), "halites": np.copy(out_halites), "ships": np.copy(out_ships), "dropoffs": np.copy(out_dropoffs)}, np.copy(out_moves[:,-1,:])
	return gen()