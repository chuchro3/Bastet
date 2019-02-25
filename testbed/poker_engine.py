import poker_constants
from poker_constants import *


class Poker():
    def __init__(self, dealer=0):
        self.deck = random.shuffle(range(52))
        self.stacks = (1000,1000)
        self.whose_turn = 0
        # 0: pre-flop, 1: flop, 2: turn, 3: river, 4: done
        self.stage = 0
        self.pot = 0
        self.round_pot = (0,0)
        self.p1_cards = self.deck[0:2]
        self.p2_cards = self.deck[2:4]
        self.common_cards = None
        self.states = [self.p1_turn, self.pot, self.p1_cards, self.common_cards]
        
    def make_action(self, action, bet_size):
        if action is FOLD:
          self.stacks[1-self.whose_turn] += self.pot
          self.stage = END_GAME
        elif action is CALL:
          chips_to_call = self.round_pot[1-self.whose_turn] - self.round_pot[self.whose_turn]
          self.stacks[player] -= chips_to_call
          self.round_pot[player] += chips_to_call
          self.pot += self.round_pot[0] + self.round_pot[1]
          self.round_pot = (0,0)
          self.stage += 1
          self.whose_turn = 0
        elif action is RAISE:
          chips_raised = bet_size - self.round_pot[self.whose_turn]
          self.round_pot[player] += chips_raised
          self.stacks[self.whose_turn] -= chips_raised
          self.whose_turn = 1 - self.whose_turn

    def get_valid_actions(self):
      # TODO
      pass

    def next_turn(self):
        if self.stage == 0:
            self.common_cards = self.deck[5:8]
        elif self.stage == 1:
            self.common_cards = self.deck[5:9]
        elif self.stage == 2:
            self.common_cards = self.deck[5:10]
        
        self.stage += 1
        return
