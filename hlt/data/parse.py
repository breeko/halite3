import copy
import json
import os
import os.path
import zstd
import numpy as np

from hlt.data.utils import get_halite, create_arr
from hlt.encoders.base import Encoder

import hlt

ARBITRARY_ID = -1

# path = "games/teccles/sample/all/ts2018-halite-3-replays_replay-20181020-192325+0000-1540063373-40-40-764391.hlt"
# player_name = "teccles"
# game = load_game(path=path, player_name="teccles")


def load_game(path: str, player_name: str, encoder: Encoder, normalize_halite: bool = True) -> dict:
	""" Loads a saved halite hlt file and extracts its halite, ships, dropoffs, cargos and moves for one player into dictionary 
		Each dictionary value will be a list with the length equal to the number of frames in a game.
		Input:
			path (str): 						path of the hlt file
			player_name (str):					name of the player to extract moves from and set relative dropoff and ships
			normalize_halite (bool): 			whether to normalize halite map to be capped at 1 based on the max halite available in the first frame
		Output:
			{
				"halite" (list(np.array)): 		halite maps by frame
				"ships" (list(np.array)): 		ship maps by frame with 1 indicating friendly ships, -1 enemy ships and 0 for empty cells
				"dropoffs" (list(np.array)): 	dropoff maps by frame with 1 indicating friendly dropoff, -1 enemy dropoff and 0 for empty cells
				"cargos" (list(dict)): 			dictionary of {ship_id: cargo} for each friendly ship
				"moves" (list(dict)): 			dictionary of {ship_id: move} for each friendly ship
			}
	"""
	max_cargo = 1000.0
	move_mappings = {"o": 0, "n": 1, "e": 2, "s": 3, "w": 4}
	
	game = parse_replay_file(file_name=path, player_name=player_name)

	out_moves     = []
	out_cargos    = []
	out_positions = []

	# game_map, moves, ships, other_ships, dropoffs, other_dropoffs
	starting_halite = get_halite(game[0][0])
	max_halite = max(np.ravel(starting_halite)) if normalize_halite else 1.0
	
	for frame in game:
	
		halite, moves, ships, other_ships, dropoffs, other_dropoffs = frame	
		
		if len(ships) == 0:
			continue

		encoded = encoder.encode(gamemap=gamemap, point=None, halite_normalize=max_halite)

		move = {ship_id: move_mappings.get(moves.get(ship_id, 0)) for ship_id in ships.keys()}
		cargo = {ship_id: ship.halite_amount / max_cargo for ship_id, ship in ships.items()}
		positions = {ship_id: ship.position for ship_id, ship in ships.items()}

		out_moves.append(move)
		out_positions.append(positions)
		out_cargos.append(cargo)
	
	# NOTE: moves, cargos and locations returned only include target player
	return {"moves": out_moves, "cargos": out_cargos, "positions": out_positions, "encoded": encoded}

def parse_replay_file(file_name, player_name, verbose=False):
	# TODO: very slow, consider re-writing
	# https://github.com/HaliteChallenge/Halite-III/blob/master/starter_kits/ml/SVM/parse.py
	if verbose:
		print("Load Replay: " + file_name)
	with open(file_name, 'rb') as f:
		data = json.loads(zstd.loads(f.read()))

	if verbose:
		print("Load Basic Information")
	player = [p for p in data['players'] if p['name'].split(" ")[0] == player_name][0]
	player_id = int(player['player_id'])
	my_shipyard = hlt.entity.Shipyard(player_id, ARBITRARY_ID,
							   hlt.Position(player['factory_location']['x'], player['factory_location']['y']))
	other_shipyards = [
		hlt.entity.Shipyard(p['player_id'], ARBITRARY_ID, hlt.Position(p['factory_location']['x'], p['factory_location']['y']))
		for p in data['players'] if int(p['player_id']) != player_id]

	if verbose:
		print("Load Cell Information")
	
	first_cells = []
	for x in range(len(data['production_map']['grid'])):
		row = []
		for y in range(len(data['production_map']['grid'][x])):
			row += [hlt.game_map.MapCell(hlt.Position(x, y), data['production_map']['grid'][x][y]['energy'])]
		first_cells.append(row)
	frames = []
	for f in data['full_frames']:
		prev_cells = first_cells if len(frames) == 0 else frames[-1]._cells
		new_cells = copy.deepcopy(prev_cells)
		for c in f['cells']:
			new_cells[c['y']][c['x']].halite_amount = c['production']
		# NOTE: Originally wrapped up in GameMap, but object only held halite cells and nothing about the ships or structures.
		# Updated to only hold a numpy array 
		frames.append(new_cells)

	if verbose:
		print("Load Player Ships")
	moves = [{} if str(player_id) not in f['moves'] else {m['id']: m['direction'] for m in f['moves'][str(player_id)] if
														  m['type'] == "m"} for f in data['full_frames']]
	ships = [{} if str(player_id) not in f['entities'] else {
		int(sid): hlt.entity.Ship(player_id, int(sid), hlt.Position(ship['x'], ship['y']), ship['energy']) for sid, ship in
		f['entities'][str(player_id)].items()} for f in data['full_frames']]

	if verbose:
		print("Load Other Player Ships")
	other_ships = [
		{int(sid): hlt.entity.Ship(int(pid), int(sid), hlt.Position(ship['x'], ship['y']), ship['energy']) for pid, p in
		 f['entities'].items() if
		 int(pid) != player_id for sid, ship in p.items()} for f in data['full_frames']]

	if verbose:
		print("Load Droppoff Information")
	first_my_dropoffs = [my_shipyard]
	first_them_dropoffs = other_shipyards
	my_dropoffs = []
	them_dropoffs = []
	for f in data['full_frames']:
		new_my_dropoffs = copy.deepcopy(first_my_dropoffs if len(my_dropoffs) == 0 else my_dropoffs[-1])
		new_them_dropoffs = copy.deepcopy(first_them_dropoffs if len(them_dropoffs) == 0 else them_dropoffs[-1])
		for e in f['events']:
			if e['type'] == 'construct':
				if int(e['owner_id']) == player_id:
					new_my_dropoffs.append(
						hlt.entity.Dropoff(player_id, ARBITRARY_ID, hlt.Position(e['location']['x'], e['location']['y'])))
				else:
					new_them_dropoffs.append(
						hlt.entity.Dropoff(e['owner_id'], ARBITRARY_ID, hlt.Position(e['location']['x'], e['location']['y'])))
		my_dropoffs.append(new_my_dropoffs)
		them_dropoffs.append(new_them_dropoffs)
	return list(zip(frames, moves, ships, other_ships, my_dropoffs, them_dropoffs))


def parse_replay_folder(folder_name, player_name, max_files=None, shuffle=False, verbose=False):
	replay_buffer = []
	file_names = sorted(os.listdir(folder_name))
	file_names = np.random.shuffle(file_names) if shuffle else file_names
	for file_name in sorted(os.listdir(folder_name)):
		if not file_name.endswith(".hlt"):
			continue
		elif max_files is not None and len(replay_buffer) >= max_files:
			break
		else:
			replay_buffer.append(parse_replay_file(os.path.join(folder_name, file_name), player_name, verbose=verbose))
	return replay_buffer

parse_replay_file("games/teccles/sample/all/ts2018-halite-3-replays_replay-20181020-192325+0000-1540063373-40-40-764391.hlt", player_name="teccles")