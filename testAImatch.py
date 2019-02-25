import pypokerengine
from players.fish_player import FishPlayer
from players.console_player import ConsolePlayer
from players.random_player import RandomPlayer
from players.emulator_player import EmulatorPlayer
from players.honest_player import HonestPlayer
from pypokerengine.api.game import setup_config, start_poker

config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=20)
#config.register_player(name="Fish", algorithm=FishPlayer())
config.register_player(name="Honest", algorithm=HonestPlayer())
#config.register_player(name="Random", algorithm=RandomPlayer())
config.register_player(name="Emulator", algorithm=EmulatorPlayer())
game_result = start_poker(config, verbose=1)
