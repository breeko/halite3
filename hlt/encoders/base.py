import importlib
from hlt.game_map import GameMap
import numpy as np


class Encoder():
	def name(self, radius: int, owner: int):
		raise NotImplementedError()
	
	def encode(self, gamemap: GameMap, point: (int, int), halite_normalize: float) -> np.array:
		raise NotImplementedError()
	
	def shape(self) -> tuple:
		raise NotImplementedError()


def get_encoder_by_name(name: str, radius: int, owner: int) -> Encoder:
	module = importlib.import_module(f"hlt.encoders.{name}")
	constructors = getattr(module, "create")
	return constructors(radius=radius, owner=owner)