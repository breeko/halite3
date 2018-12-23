import numpy as np
from hlt.positionals import Position
from hlt.game_map import GameMap
from hlt.encoders import roll_and_crop

def one_hot(arr: list, num_classes: int) -> np.array:
	""" Turns a list of integers into a one hot array"""
	assert max(arr) < num_classes
	out = np.zeros(shape=[len(arr), num_classes])
	for idx, val in enumerate(arr):
		out[idx][val] = 1.0
	return np.array(out)

def create_arr(positions: {int: Position}, shape: [int]) -> list:
	""" Takes a dictionary of positions and plots them on an array of zeros """
	ships = np.zeros(shape)
	for k, pos in positions.items():
		for p in pos:
			ships[p.y, p.x] = k
	return ships

def get_halite(gamemap: GameMap) -> np.array:
	"""" returns an numpy array of halite from the gamemap"""
	halite = []
	cells = gamemap._cells
	for r in cells:
		row = []
		for val in r:
			row.append(val.halite_amount)
		halite.append(row)
	return np.array(halite)

def stack_relative_frame(game: dict, ship_id: int, frame: int, look_back: int, radius: int) -> dict:
	""" Orients a game dictionary frame into a frame centered around a given ship
		inputs:
			game (dict): 	 dictionary containing keys: ["halite", "ships", "dropoffs", "moves", "cargos"]
			ship_id (int):   id of the ship that will be in the center of all maps
			frame (int):     which frame number to start with
			look_back (int): how many frames to look back. e.g. lookback of 1, will return 2 frames (latest and prior to latest)
			radius (int):	 how far in each direction to look out when cropping the maps
		output:
			out (dict):		 dictionary of {halite, ships, dropoffs, moves, cargos} all with the ship specified by the ship_id in the center
	"""
	start_frame = max(0, frame - look_back)
	end_frame = frame

	map_shape = (radius * 2 + 1, radius * 2 + 1)
	empty = np.zeros(shape=map_shape)

	missing_frames = look_back - (end_frame - start_frame)
	out_halite    = [empty for _ in range(missing_frames)]
	out_ships     = [empty for _ in range(missing_frames)]
	out_dropoffs  = [empty for _ in range(missing_frames)]
	out_moves     = [0 for _ in range(missing_frames)]
	out_cargos    = [0.0 for _ in range(missing_frames)]

	for frame in range(start_frame, end_frame + 1):
		position = game["positions"][frame].get(ship_id)
		if position is None:
			out_halite.append(empty)
			out_ships.append(empty)
			out_dropoffs.append(empty)
			out_moves.append(0)
			out_cargos.append(0.0)
		else:
			rel_halite = roll_and_crop(arr=game["halite"][frame], point=position, radius=radius)
			rel_ships = roll_and_crop(arr=game["ships"][frame], point=position, radius=radius)
			rel_dropoffs = roll_and_crop(arr=game["dropoffs"][frame], point=position, radius=radius)
			out_halite.append(rel_halite)
			out_ships.append(rel_ships)
			out_dropoffs.append(rel_dropoffs)
			out_moves.append(game["moves"][frame].get(ship_id) or 0) # NOTE: game["moves"][frame].get(ship_id) may return None so use default outside of get
			out_cargos.append(game["cargos"][frame].get(ship_id) or 0.0)

	return {"halites": np.stack(out_halite, axis=-1), "ships": np.stack(out_ships,axis=-1), "dropoffs": np.stack(out_dropoffs, axis=-1), "moves": out_moves, "cargos": out_cargos}
