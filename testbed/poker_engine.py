import poker_constants
from poker_constants import *
import random


class Poker():
    def __init__(self, dealer=0):
        self.deck = list(range(52))
        random.shuffle(self.deck)
        self.stacks = [INITIAL_STACK, INITIAL_STACK]
        self.last_actions = [None, None]
        self.reward = [0, 0]
        self.whose_turn = 0
        self.stage = 0
        self.next_stage_ready = 0
        self.pot = 0
        self.round_pot = [0,0]
        self.hole_cards = [None, None]
        self.common_cards = [None] * 5
        self.valid_actions = ACTIONS
        self.states = [self.whose_turn, self.stacks, self.pot, self.hole_cards[0], self.common_cards,
                       self.stage, self.last_actions]
    
    def get_states(self):
        return [self.whose_turn, self.stacks, self.pot, self.hole_cards[0], self.common_cards,
                       self.stage, self.last_actions]
    
    def deal_cards(self):
        self.stage += 1
        self.hole_cards = (self.deck[0:2], self.deck[2:4])
        self.pot = SMALL_BLIND_AMOUNT * 3
        self.stacks[self.whose_turn] -= SMALL_BLIND_AMOUNT
        self.stacks[1-self.whose_turn] -= SMALL_BLIND_AMOUNT*2
        self.whose_turn = 0
        self.round_pot = [0, SMALL_BLIND_AMOUNT]
        self.next_stage_ready = 0
        self.states = self.get_states()
        
        
    def make_action(self, action, bet_size=None):
        self.last_actions[self.whose_turn] = action
        
        if action is FOLD:
          self.stacks[1-self.whose_turn] += self.pot
          # update reward
          self.reward[0] = self.stack[0] - INITIAL_STACK
          self.reward[1] = self.stack[1] - INITIAL_STACK
          self.stage = END_GAME
          
        elif action is CHECK:
          # update reward
          self.reward[self.whose_turn] = 0
          
          # decide whether to move on next turn
          if self.next_stage_ready:
              self.next_turn()
          else:
              self.whose_turn = 1 - self.whose_turn
              self.next_stage_ready = 1
          
        elif action is CALL:
          chips_to_call = self.round_pot[1-self.whose_turn] - self.round_pot[self.whose_turn]
          # update reward
          self.reward[self.whose_turn] = - chips_to_call
          # update stack
          self.stacks[self.whose_turn] -= chips_to_call
          self.round_pot[self.whose_turn] += chips_to_call
          self.pot += self.round_pot[0] + self.round_pot[1]
          
          # move on to next turn
          self.next_turn()
          
          
        elif action is RAISE:
          if bet_size == None:
              bet_size == min(self.pot, self.stacks[whose_turn])
          
          chips_raised = bet_size - self.round_pot[self.whose_turn]
          # update reward
          self.reward[self.whose_turn] = - chips_raised

          self.round_pot[self.whose_turn] += chips_raised
          self.pot += chips_to_call
          self.stacks[self.whose_turn] -= chips_raised
          self.whose_turn = 1 - self.whose_turn
          self.valid_actions = [CALL, FOLD, RAISE, ALL_IN]
        
        elif action is ALL_IN:
          chips_to_call = self.stacks[self.whose_turn]
          self.stacks[self.whose_turn] = 0
          self.round_pot[self.whose_turn] += chips_to_call
          self.pot += chips_to_call
          # update reward
          self.reward[self.whose_turn] = - chips_to_call
          
          self.whose_turn = 1 - self.whose_turn
          self.valid_actions = [CALL, FOLD, ALL_IN]
        self.states = self.get_states()

    def get_valid_actions(self):
      # TODO
      pass

    def next_turn(self):
        self.stage += 1
        self.round_pot = [0,0]
        self.whose_turn = 0
        
        if self.stage == PREFLOP:
            self.common_cards = [None] * 5
        if self.stage == FLOP:
            self.common_cards = self.deck[5:8] + [None, None]
        elif self.stage == TURN:
            self.common_cards = self.deck[5:9] + [None]
        elif self.stage == RIVER:
            self.common_cards = self.deck[5:10]
        elif self.stage == SHOW_HAND:
            ## TODO by Nick
            pass
        elif self.stage == END_GAME:
            pass
        self.states = self.get_states()
    

if __name__ == '__main__':
    poker = Poker()
    print(poker.states)
    poker.deal_cards()
    print(poker.states)
    poker.make_action(CALL)
    print(poker.states)
    poker.make_action(CHECK)
    print(poker.states)

    poker.make_action(CHECK)
    print(poker.states)
    
    rewards = poker.reward[0]
    # other player, in this case fish, makes another action
    # if game hasn't ended.
    self.poker.make_action(CALL, 0)

