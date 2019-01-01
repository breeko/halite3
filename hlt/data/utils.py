import numpy as np
from hlt.positionals import Position
from hlt.game_map import GameMap

def get_move_counts(player: str, frames_moves: list, relative: True) -> list:
	""" Takes a list of frame moves by player and ship-id and returns a count of each move type	"""
	player_moves = {}
	for frame_moves in frames_moves:
		frame_player_moves = frame_moves.get(player)
		if frame_player_moves:
			for player_move in frame_player_moves.values():
				player_moves[player_move] = player_moves.get(player_move, 0) + 1
	if relative:
		sum_vals = sum(player_moves.values())
		player_moves = {k: v / float(sum_vals) for k,v in player_moves.items()}
	return player_moves

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

def get_rotated_direction(move: str, num_rotations: int):
	""" Returns the relative direction of a move after a series of 90-degree counter-clockwise rotations 
		e.g. 
			0,1,2	1-rot	2,5,8
			3,4,5	 ->	 	1,4,7
			6,7,8			0,3,6

	"""
	rotation_mapping = {
			"n": "w",
			"w": "s",
			"s": "e",
			"e": "n",
			"o": "o"
		}
	cur_move = move
	for _ in range(num_rotations):
		cur_move = rotation_mapping.get(cur_move)
	return cur_move