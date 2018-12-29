import importlib
import numpy as np
import json

class Encoder():
	def name(self):
		raise NotImplementedError()
	
	def encode_from_file(self, path: str) -> None:
		with open(path,"r") as f:
			raw = f.read()
			game = json.loads(raw)
		return self.encode_from_dict(game)

	def encode_from_dict(self, game: dict) -> None:
		raise NotImplementedError()
	
	def encode_from_gamemap(self, game) -> None:
		raise NotImplementedError()

def get_encoder_by_name(name: str) -> Encoder:
	module = importlib.import_module("hlt.encoders.{}".format(name))
	constructors = getattr(module, "create")
	return constructors()