import os
import numpy as np
from hlt.data.utils import one_hot, create_arr, get_move_counts, get_rotated_direction
from hlt.encoders.base import get_encoder_by_name
from hlt.encoders.utils import roll_and_crop

class Generator:
	def __init__(
		self,
		encoder_name: str,
		replay_folder: str, 
		player_name: str,
		radius: int,
		prob_include_frame:float=0.2,
		prob_include_ship:float=0.2,
		batch_size:int=128,
		start_frame_perc:float=0.0,
		end_frame_perc:float=1.0,
		equal_move_prob:bool=True,
		rotate:bool=True,) -> (dict, np.array):
		""""
			Input generator for training a neural network
			inputs:
				encoder_name (str):							name of the encoder to use
				replay_folder (str): 						directory that stores .hlt games
				player_name (str): 							name of the player you want to create a training set for
				radius (int): 								how many squares to consider in each direction
				prob_include_frame (float - default 0.2): 	probability to include a frame when aggregating frames 
				prob_include_ship (float - default 0.2):  	probability to include a ship when aggregating ships from frames 
				batch_size (int - default 128):				size of the training batch
				start_frame_perc (float - default 0.0):   	the frame percent to start on (e.g. 0.2 means start 20% through the game)	
				end_frame_perc (float - default 1.0):     	the frame percent to start on (e.g. 0.9 means end after 90% of game is through)
				equal_move_prob (boolean):					if True, this provided an equal sampling of all possible moves
			
			outputs:
				[{"maps", "move_costs", "halites", "ships", "dropoffs", "cargos"}, outs]

				maps:	  	(None, (2 * radius + 1), (2 * radius + 1), 4)	concatenation of relative and normalized [halite, ships, structures, move_costs]
				move_costs: (None, (2 * radius + 1), (2 * radius + 1), 1)	how many time current ship can move on a given square
				halites:   	(None, (2 * radius + 1), (2 * radius + 1), 1), 	halite ship is carrying
				ships:    	(None, (2 * radius + 1), (2 * radius + 1), 1), 	ships on map, 1 representing friendly ship and -1 representing enemy ship
				dropoffs: 	(None, (2 * radius + 1), (2 * radius + 1), 1), 	dropoffs on map, 1 representing friendly dropoff and -1 representing enemy dropoff
				cargos:   	(None, 1),                                 		cargo ship is carrying
				outs:     	(None, 5),			   			                one-hot vector of move (north, south, east, west, still)
				
		"""
		# TODO: Make player_name into a lambda to allow for things like winning player
		# TODO: Implement lookback
		
		# user defined specs
		self.replay_folder = replay_folder
		self.player_name = player_name
		self.radius = radius
		self.prob_include_frame = prob_include_frame
		self.prob_include_ship = prob_include_ship
		self.batch_size = batch_size
		self.start_frame_perc = start_frame_perc
		self.end_frame_perc = end_frame_perc

		self.encoder_name = encoder_name
		self.encoder = get_encoder_by_name(encoder_name)

		self.move_mapping = {"n": 0, "s": 1, "e": 2, "w": 3, "o": 4}
		self.num_move_types = len(self.move_mapping)
		self.equal_move_prob = equal_move_prob
		self.rotate = rotate
	
	@property
	def output_shape(self):
		return (self.radius * 2 + 1, self.radius * 2 + 1, 4)
		
	def __next__(self):
		# TODO: encode inspired pane
		map_shape = self.radius * 2 + 1, self.radius * 2 + 1
		num_classes = 5

		out_cargos   	= np.zeros(shape=(self.batch_size, 1), dtype=np.float)
		out_moves    	= np.zeros(shape=(self.batch_size, num_classes), dtype=np.float)
		out_halites  	= np.zeros(shape=(self.batch_size, *map_shape, 1), dtype=np.float)
		out_ships    	= np.zeros(shape=(self.batch_size, *map_shape, 1), dtype=np.float)
		out_dropoffs 	= np.zeros(shape=(self.batch_size, *map_shape, 1), dtype=np.float)
		out_move_costs 	= np.zeros(shape=(self.batch_size, *map_shape, 1), dtype=np.float)
		out_maps	 	= np.zeros(shape=(self.batch_size, *map_shape, 4), dtype=np.float)

		ct = 0
		available_files = [f for f in os.listdir(self.replay_folder) if f.endswith(".json")]

		while True:
			file_name = np.random.choice(available_files, size=1)[0]
			file_path = "{}/{}".format(self.replay_folder, file_name)
			
			encoded = self.encoder.encode_from_file(path=file_path)

			player_id = encoded["players"][self.player_name]
			num_frames = encoded["num_frames"]
			constants = encoded["constants"]

			move_cost_ratio = float(constants["MOVE_COST_RATIO"])

			max_cell_production = float(constants["MAX_CELL_PRODUCTION"])
			frame_shape = encoded["halites"][0].shape

			move_counts = get_move_counts(player=player_id, frames_moves=encoded["moves"], relative=True)
			min_move_prob = min(move_counts.values())
			move_probs = {k: min_move_prob / float(v) for k, v in move_counts.items()}
			
			for num_frame in range(num_frames):
				perc_frame = num_frame / float(num_frames)
				if perc_frame < self.start_frame_perc:
					continue
				elif perc_frame > self.end_frame_perc:
					break
				
				if np.random.random() > self.prob_include_frame:
					continue

				frame_halites = encoded["halites"][num_frame]
				frame_moves = encoded["moves"][num_frame]
				frame_ships = encoded["ships"][num_frame]
				frame_structures = encoded["structures"][num_frame]

				player_moves = frame_moves.get(str(player_id))
				
				if player_moves is None:
					continue
								
				arr_ships = create_arr(locations=frame_ships, player=player_id, shape=frame_shape)
				arr_structures = create_arr(locations=frame_structures, player=player_id, shape=frame_shape)

				for ship_id, move in player_moves.items():
					if np.random.random() > self.prob_include_ship:
						continue

					if self.equal_move_prob and np.random.random() > move_probs[move]:
						continue

					ship = frame_ships[player_id][ship_id]
					cargo = ship["energy"]

					x = ship["x"]
					y = ship["y"]

					rel_halites 	= roll_and_crop(arr=frame_halites, x=x, y=y, radius=self.radius)
					rel_ships 		= roll_and_crop(arr=arr_ships, x=x, y=y, radius=self.radius)
					rel_structures 	= roll_and_crop(arr=arr_structures, x=x, y=y, radius=self.radius)
					rel_move_costs  = cargo - (rel_halites / move_cost_ratio) # how many times you can move from this space

					 # normalize
					norm_rel_halites 	 = rel_halites / max_cell_production
					norm_rel_move_costs  = rel_move_costs / max_cell_production

					if self.rotate:
						num_rotations = np.random.randint(4) # 0 - 4
						new_move = get_rotated_direction(move, num_rotations)
						move = new_move

						rel_halites = np.rot90(rel_halites, k=num_rotations)
						rel_ships = np.rot90(rel_ships, k=num_rotations)
						rel_structures = np.rot90(rel_structures, k=num_rotations)
						rel_move_costs = np.rot90(rel_move_costs, k=num_rotations)
					
					rel_move 			= one_hot(arr=move, num_classes=self.num_move_types, mapping=self.move_mapping)

					out_halites[ct]  	= rel_halites
					out_ships[ct]    	= rel_ships
					out_dropoffs[ct] 	= rel_structures
					out_moves[ct]    	= rel_move
					out_move_costs[ct] 	= rel_move_costs
					out_maps[ct]	 	= np.stack([norm_rel_halites, rel_ships, rel_structures, norm_rel_move_costs], axis=-1).squeeze()
					out_cargos[ct]		= cargo

					ct = (ct + 1) % self.batch_size
					
					if ct == 0:
						inputs = {	"maps": np.copy(out_maps),
									"move_costs": np.copy(out_move_costs),
									"cargos": np.copy(out_cargos),
									"halites": np.copy(out_halites),
									"ships": np.copy(out_ships),
									"dropoffs": np.copy(out_dropoffs)}
						outputs = np.copy(out_moves)
						return inputs, outputs 