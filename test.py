from hlt.models.small import get_model
from hlt.data.generator import Generator
from hlt_client.download_game import download
from hlt_client.client import REPLAY_MODE_USER
from hlt.encoders.base import get_encoder_by_name

import numpy as np


train_folder = "hlt/games/train/"
val_folder = "hlt/games/val"
player_name = "teccles"

# game_file = "hlt/games/ts2018-halite-3-gold-replays_replay-20181022-232058+0000-1540250304-64-64-810977.json"
# encoder = get_encoder_by_name("historic")
# encoded = encoder.encode_from_file(game_file)

train_gen = Generator(player_name=player_name, encoder_name="historic", replay_folder=train_folder, radius=10)
val_gen = Generator(player_name=player_name, encoder_name="historic", replay_folder=val_folder, radius=10)
(a,b) = next(val_gen)

map_shape = (21,21,4)
model = get_model(map_shape)

model.summary()
help(model.fit_generator)
model.fit_generator(generator=train_gen, validation_data=val_gen, validation_steps=10,steps_per_epoch=50, epochs=20)
np.argmax(model.predict(a), axis=1)
np.argmax(b, axis=1)

np.sum(b,axis=0)
a["maps"][0,:,:,-1]
a["ships_halite"][0]
b
# download(
#     mode=REPLAY_MODE_USER,
#     destination="hlt/games",
#     date=None,
#     all_bots=None,
#     default_user_id=6416,
#     user_id=2807,
#     limit=1000,
#     decompress=True)
