import pypokerengine
import pprint
from players.deep_player import DeepPlayer
from players.console_player import ConsolePlayer
from players.random_player import RandomPlayer
from pypokerengine.api.game import setup_config, start_poker

config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=20)
config.register_player(name="Deep", algorithm=DeepPlayer())
config.register_player(name="Random", algorithm=RandomPlayer())
game_result = start_poker(config, verbose=1)
print("\nGAME RESULT\n")
pprint.PrettyPrinter().pprint(game_result)

