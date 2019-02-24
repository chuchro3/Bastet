from pprint import PrettyPrinter
from pypokerengine.players import BasePokerPlayer

pp = PrettyPrinter()

# A skeleton for allowing our trained DeepPlayer to interact with PyPokerEngine.
class DeepPlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        print()
        print("VALID_ACTIONS:")
        pp.pprint(valid_actions)
        print("\nHOLE_CARD:")
        pp.pprint(hole_card)
        print("\nROUND_STATE:")
        pp.pprint(round_state)
        print("\nEXTRACTED FEATURES LEN:", len(Model().extract_features(hole_card, round_state)))
        print()
        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

# utility methods for extracting features --------------------------------------

# returns position (button = 0,  first to act = 1)
def get_position_feature(round_state):
    return [1 if round_state['dealer_btn'] == round_state['next_player'] else 0]

def get_hole_cards_feature(hole_cards):
  cards = [0] * 52
  for suit,card in hole_cards:
    try:
      suit_value = {
        'C': 0,
        'D': 1,
        'H': 2,
        'S': 3
      }[suit]
    except:
      raise Exception("Invalid Suit: " + suit)
    try:
      card_value = {
        'A': 0,
        '2': 1,
        '3': 2,
        '4': 3,
        '5': 4,
        '6': 5,
        '7': 6,
        '8': 7,
        '9': 8,
        'T': 9,
        'J': 10,
        'Q': 11,
        'K': 12,
      }[card]
    except:
      raise Exception("Invalid Card: " + card)
    cards[suit_value*13 + card_value] = 1
  return cards

# The meat of our CS 234 Poker Bot.
class Model():
  # Takes info from PyPokerEngine objects, and extracts the features we want.
  def extract_features(self, hole_card, round_state):
    features = []
    features += get_position_feature(round_state)
    features += get_hole_cards_feature(hole_card)
    return features
