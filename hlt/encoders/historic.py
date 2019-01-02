from hlt.encoders.base import Encoder
from hlt.networking import Game 
import numpy as np
from copy import deepcopy

class HistoricEncoder(Encoder):
    def __init__(self):
        pass
    
    def encode_from_gamemap(self, game: Game) -> None:
        # TODO: fill me out
        raise NotImplementedError()
                 
    def _get_initial_halite(self, production_map: dict, width: int, height: int) -> np.array:
        grid = production_map["grid"]
        halite = np.zeros(shape=(width, height, 1))
        for r_idx, row in enumerate(grid):
            for c_idx, val in enumerate(row):
                halite[r_idx, c_idx, 0] = float(val["energy"])
        return halite

    def _get_initial_structure(self, players: [dict]) -> np.array:
        starting_structure = {}
        for player in players:
            factory_location = player["factory_location"]
            y = factory_location["y"]
            x = factory_location["x"]
            player_id = str(player["player_id"])
            starting_structure[player_id] = {"0": {"x": x, "y": y}}
        return starting_structure

    def encode_from_dict(self, historic: [dict]) -> None:
        """
            Returns encoded game. Variables saved:
                energies: list of how much energy each player has per turn
                moves: list of moves e.g. {ship_id: [{'direction': 'n', 'id': 16, 'type': 'm'},...]}
                halites: list of map of halite per frame. shape: [num_frames, map_height, map_width, 1]
                structures: list of map of structures per frame  e.g. {owner: {structure_id: {"x": x, "y": y}}}
                ships: list of map of ships per frame

            inputs:
                frames list(dict): a list of dictionary frames from save game json (e.g. game["full_frames"])
            outputs:
                None           
        """
        # TODO: handle generate
        production_map = historic["production_map"]
        players = historic["players"]
        frames = historic["full_frames"]
        constants = historic["GAME_CONSTANTS"]
        
        width = production_map["width"]
        height = production_map["height"]
        
        prior_halite = self._get_initial_halite(production_map=production_map, width=width, height=height)
        prior_structure = self._get_initial_structure(players=players)

        player_names = {" ".join(p["name"].split()[:-1]): str(p["player_id"]) for p in players}
        
        # list of dictionaries
        energies = []       # [{owner: halite}, ...]                # NOTE: does not include ship's halites
        moves = []          # [{owner: {ship_id: direction}}, ...]  # NOTE: directions are 'n','s','e','w','o'
        structures = []     # [{owner: [{"x", "y"},...], ...] 
        ships = []          # [{owner: {ship_id: {"x", "y", "energy"}},...]
        
        # map
        halites = []        # [num_frames, height, width, 1]

        for frame in frames:
            # TODO: deal with g (generate) moves

            # energy
            frame_energy = frame["energy"]
            energies.append(frame_energy)

            # halite
            frame_cells = frame["cells"]
            cur_halite = np.copy(prior_halite)
            for cell in frame_cells:
                c = cell["x"]
                r = cell["y"]
                production = cell["production"]
                cur_halite[r,c,0] = float(production)
            halites.append(cur_halite)
            prior_halite = cur_halite
            
            # structures = [{owner: {structure_id: {"x", "y"}}, ...] 
            frame_events = frame["events"]
            cur_structure = deepcopy(prior_structure)
            for event in frame_events:
                if event["type"] == "construct":
                    y = event["location"]["y"]
                    x = event["location"]["x"]
                    owner = str(event["owner_id"])
                    structure_id = str(event["id"])
                    if owner not in prior_structure:
                        cur_structure[owner] = {}
                    cur_structure[owner][structure_id] = {"x": x, "y": y}
            structures.append(cur_structure)

            # ships = [{owner: {ship_id: {"x", "y", "energy"}}}, ...]
            # moves = [{owner: {ship_id: direction}}, ...]
            frame_moves = frame["moves"]

            cur_moves = {
                str(owner_id): {
                    str(move["id"]): move["direction"] for move in moves if move["type"] == "m"} for owner_id, moves in frame_moves.items()}
            
            cur_ships = frame["entities"]

            # add ships that didn't explicitly move
            for owner, owner_ships in cur_ships.items():
                for ship_id in owner_ships.keys():
                    if owner not in cur_moves:
                        cur_moves[owner] = {}
                    if ship_id not in cur_moves[owner]:
                        cur_moves[owner][str(ship_id)] = "o"
            
            moves.append(cur_moves)
            ships.append(cur_ships)

        # dictionaries
        return {    
            "halites":      np.stack(halites,axis=0),
            "energies":     energies,
            "moves":        moves,
            "structures":   structures,
            "ships":        ships,
            "num_frames":   len(frames),
            "players":      player_names,
            "constants":    constants
        }
    
    @property
    def name(self) -> str:
        return "historic"

def create():
    return HistoricEncoder()