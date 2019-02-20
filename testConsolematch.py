import pypokerengine
from players.fish_player import FishPlayer
from players.console_player import ConsolePlayer
from players.random_player import RandomPlayer
from pypokerengine.api.game import setup_config, start_poker

config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=20)
config.register_player(name="Fish", algorithm=FishPlayer())
config.register_player(name="Human", algorithm=ConsolePlayer())
game_result = start_poker(config, verbose=1)
