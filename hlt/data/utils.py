import numpy as np
from hlt.positionals import Position
from hlt.game_map import GameMap

def one_hot(arr: list, num_classes: int, mapping: dict = {}) -> np.array:
	""" Turns a list of integers into a one hot array"""
	if type(arr) is not list:
		arr = [arr]
	out = np.zeros(shape=[len(arr), num_classes])
	for idx, val in enumerate(arr):
		mapped_val = mapping.get(val, val)
		out[idx][mapped_val] = 1.0
	return np.array(out)

def create_arr(locations: dict, player: str, shape: [int], player_key: str = 1, other_key: str = -1) -> list:
	""" Takes a dictionary of locations and plots them on an array of zeros with player_key as locations
		holding the player and other_key for locations holding non-player
	"""
	ships = np.zeros(shape)
	for k, pos in locations.items():
		for p in pos.values():
			map_key = player_key if k == player else other_key
			ships[p["y"]][p["x"]] = map_key
	return ships
