import numpy as np 

from hlt.encoders.base import Encoder
from hlt.encoders.utils import roll_and_crop
from hlt.game_map import GameMap

class ThreePlaneEncoder(Encoder):
	def __init__(self, radius: int, owner: int):
		self.board_width  = radius * 2 + 1
		self.board_height = radius * 2 + 1
		self.radius = radius
		self.owner = owner
		self.num_planes = 3
	
	def name(self) -> str:
		return "threeplane"

	def encode(self, gamemap: GameMap, point: (int, int) = None, halite_normalize: float = None) -> np.array:
		""" Converts a gamemap object into a three plane board centered around point 
			inputs:
				gamemap (GameMap): 	game object representing the current game state
				point (int, int):  	2-d point used to center the board around
			outputs:
				arr (np.array):		numpy array representing halite, ships and structures, shape of (3, radius * 2 + 1, radius * 2 + 1)
		"""
		shifted_cells = roll_and_crop(arr=gamemap._cells, point=point, radius=self.radius)
		
		halite 		= self.encode_halites(cells=shifted_cells)
		ships 		= self.encode_ships(cells=shifted_cells)
		structures 	= self.encode_structures(cells=shifted_cells)

		for idx, cell in enumerate(shifted_cells.flatten()):
			halite[idx] = cell.halite_amount
			if cell.ship is not None:
				ship = 1 if cell.ship.owner == self.owner else -1
				ships[idx] = ship
			if cell.structure is not None:
				structure = 1 if cell.structure.owner == self.owner else -1
				structures[idx] = structure
	
		if halite_normalize:
			halite = halite / float(halite_normalize)

		return np.stack([halite, ships, structures], axis=0).reshape([self.num_planes,self.board_height,self.board_width])
	
	def encode_halites(self, cells: np.array) -> np.array:
		halite = np.zeros(shape=(self.board_width * self.num_planes))
		for idx, cell in enumerate(cells.flatten()):
			halite[idx] = cell.halite_amount
		return halite.reshape(self.board_height, self.board_width)
	
	def encode_ships(self, cells: np.array) -> np.array:
		ships = np.zeros(shape=(self.board_width * self.num_planes), dtype=np.uint8)
		for idx, cell in enumerate(cells.flatten()):
			if cell.ship is not None:
				ship = 1 if cell.ship.owner == self.owner else -1
				ships[idx] = ship
		return ships.reshape(self.board_height, self.board_width)
	
	def encode_structures(self, cells: np.array) -> np.array:
		structures = np.zeros(shape=(self.board_width * self.num_planes), dtype=np.uint8)
		for idx, cell in enumerate(cells.flatten()):
			if cell.structure is not None:
				ship = 1 if cell.ship.owner == self.owner else -1
				structures[idx] = ship
		return structures.reshape(self.board_height, self.board_width)
	
	def shape(self) -> (int, int, int):
		""" Returns board shape """
		return (self.num_planes, self.board_height, self.board_width)

