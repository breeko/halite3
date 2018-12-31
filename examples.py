from hlt.models.small import get_model
from hlt.data.generator import Generator
from hlt_client.download_game import download
from hlt_client.client import REPLAY_MODE_USER
from hlt.encoders.base import get_encoder_by_name

import numpy as np

# Download games
my_user_id = 6416
download_user_id = 2807
limit = 1000
download_path = "hlt/games"
download(
    mode=REPLAY_MODE_USER,
    destination=download_path,
    date=None,
    all_bots=None,
    default_user_id=my_user_id,
    user_id=download_user_id,
    limit=limit,
    decompress=True)

# Encode a single game
game_file = "hlt/games/sample/ts2018-halite-3-gold-replays_replay-20181020-201720+0000-1540066636-48-48-765185.json"
encoder = get_encoder_by_name("historic")
encoded = encoder.encode_from_file(game_file)
encoded.keys() # dict_keys(['structures', 'players', 'moves', 'ships', 'num_frames', 'constants', 'energies', 'halites'])

# Create a generator
player_name = "teccles"
radius = 2
sample_folder = "hlt/games/sample"
sample_gen = Generator(
    player_name=player_name,
    batch_size=128,
    encoder_name="historic",
    replay_folder=sample_folder,
    radius=radius)

# Review output of a generator
(inp, out) = next(sample_gen)
inp["maps"].shape # 128, 5, 5, 4 => batch_size, radius * 2 + 1, radius * 2 + 1, 4

# NOTE: Since a generator returns an input that is a map, your model input has to reference the parameter it uses by name
# For instance, if you want to run a convolutional layer on maps, you have to define it by Input(..., name="maps")
# Other keys will be ignored by the model

# Four maps are halite, ships, structures, move_cost
halite_index = 0
ships_index = 1
structures_index = 2
move_cost_index = 3

# All maps are relative with the ship being located in the center.
# For instance, if radius is 2, there are 2 map cells to the left, right, above and below.
# With a radius of 2, the ship will always be in index 2,2
inp["maps"][:,radius,radius,ships_index] # always 1.

# halites are relative scaled based on the maximum amount of halite available in the first frame
np.max(inp["maps"][:,:,:,halite_index]) # < 1.0
np.min(inp["maps"][:,:,:,halite_index]) # >= 0.0

# move_costs are scaled based on how many times a player can move on each square
np.max(inp["maps"][:,:,:,move_cost_index]) # may be > 1.0
np.min(inp["maps"][:,:,:,move_cost_index]) # may be < 0.0

# Moves will be one hot encoded based on move_mapping
sample_gen.move_mapping
out.shape   # batch_size, 5

# Create a model
model = get_model(sample_gen.output_shape)
model.summary()

# Train a model
model.fit_generator(generator=sample_gen, steps_per_epoch=1)
