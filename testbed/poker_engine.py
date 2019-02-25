import poker_constants
from poker_constants import *
import random


class Poker():
    def __init__(self, dealer=0):
        # card initialization
        self.deck = list(range(52))
        random.shuffle(self.deck)
        self.hole_cards = (self.deck[0:2], self.deck[2:4])
        self.common_cards = [None] * 5

        # game state initialization
        self.last_actions = [None, None]
        self.reward = [0, 0]
        self.whose_turn = 0
        self.stage = 0
        self.next_stage_ready = 0
        self.valid_actions = ACTIONS

        # chip initialization
        self.stacks = [INITIAL_STACK, INITIAL_STACK]
        self.pot = 0
        self.round_pot = [0,0]

#def deal_cards(self):
        # when it's preflop, dealer (player0) is BB and player1 is first to act
        self.stage = PREFLOP
        self.pot = SMALL_BLIND_AMOUNT * 3
        self.stacks[0] -= 2*SMALL_BLIND_AMOUNT
        self.stacks[1] -= SMALL_BLIND_AMOUNT
        self.whose_turn = 1
        self.round_pot = [2*SMALL_BLIND_AMOUNT, SMALL_BLIND_AMOUNT]
        self.next_stage_ready = 0

    # player is either 0 or 1
    def get_state(self, player):
        return [self.whose_turn, self.stacks, self.pot, self.hole_cards[player], self.common_cards,
                       self.stage, self.last_actions]
        
    def make_action(self, action, bet_size=None):
        self.last_actions[self.whose_turn] = action
        
        if action is FOLD:
          self.stacks[1-self.whose_turn] += self.pot
          # update reward
          self.reward[0] = self.stack[0] - INITIAL_STACK
          self.reward[1] = self.stack[1] - INITIAL_STACK
          self.stage = END_GAME
          
        elif action is CHECK:
          # ------
          # edit by nick: commented out the below. why update reward in check?
          # update reward
          # self.reward[self.whose_turn] = 0
          # ------
          
          # decide whether to move on next turn
          if self.next_stage_ready:
              self.next_turn()
          else:
              self.whose_turn = 1 - self.whose_turn
              self.next_stage_ready = 1
          
        elif action is CALL:
          chips_to_call = self.round_pot[1-self.whose_turn] - self.round_pot[self.whose_turn]
          # ------
          # edit by nick: commented out the below. why update reward in check?
          # update reward
          # self.reward[self.whose_turn] = - chips_to_call
          # ------
          # update stack
          self.stacks[self.whose_turn] -= chips_to_call
          self.round_pot[self.whose_turn] += chips_to_call
          self.pot += self.round_pot[0] + self.round_pot[1]
          
          # move on to next turn if round is done (in preflop, BB can raise)
          if self.next_stage_ready:
            self.next_turn()
          else:
            self.next_stage_ready = 1
          
          
        elif action is RAISE:
          if bet_size == None:
              bet_size = min(self.pot, self.stacks[whose_turn])
          
          chips_raised = bet_size - self.round_pot[self.whose_turn]
          # -----
          # update reward
          # edit by nick: commented out the below. why update reward in raise?
          # self.reward[self.whose_turn] = - chips_raised
          # -----

          self.round_pot[self.whose_turn] += chips_raised
          self.pot += chips_raised
          self.stacks[self.whose_turn] -= chips_raised
          self.whose_turn = 1 - self.whose_turn
        
        elif action is ALL_IN:
          chips_to_call = self.stacks[self.whose_turn]
          self.stacks[self.whose_turn] = 0
          self.round_pot[self.whose_turn] += chips_to_call
          self.pot += chips_to_call
          # update reward
          #self.reward[self.whose_turn] = - chips_to_call
          
          self.whose_turn = 1 - self.whose_turn

    def get_valid_actions(self):
      # TODO
      pass

    def next_turn(self):
        self.stage += 1
        self.round_pot = [0,0]
        self.whose_turn = 0
        self.next_stage_ready = 0
        
        if self.stage == PREFLOP:
            self.common_cards = [None] * 5
        elif self.stage == FLOP:
            self.common_cards = self.deck[5:8] + [None, None]
        elif self.stage == TURN:
            self.common_cards = self.deck[5:9] + [None]
        elif self.stage == RIVER:
            self.common_cards = self.deck[5:10]
        elif self.stage == SHOW_HAND:
            # TODO: update, hardcode p1 to always win for now
            self.stacks[0] += self.pot
            # update reward
            self.reward[0] = self.stack[0] - INITIAL_STACK
            self.reward[1] = self.stack[1] - INITIAL_STACK
        elif self.stage == END_GAME:
            pass
    

if __name__ == '__main__':
    poker = Poker()
    print(poker.get_state(0))
    poker.make_action(CALL)
    print(poker.get_state(0))
    poker.make_action(CHECK)
    print(poker.get_state(0))

    poker.make_action(CHECK)
    print(poker.get_state(0))
    
    rewards = poker.reward[0]
    # other player, in this case fish, makes another action
    # if game hasn't ended.
    poker.make_action(CALL, 0)

