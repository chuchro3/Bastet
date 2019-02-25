import random

class Poker():
  def __init__(self):
    self.deck = random.shuffle(range(52))
    self.p1_stack = 1000
    self.p2_stack = 1000
    self.p1_turn = True
    
  def get_p1_cards(self):
    return self.deck[0:2]

  def get_p2_cards(self):
    return self.deck[2:4]

  def get_flop(self):
    return self.deck[5:8]

  def get_turn(self):
    return self.deck[5:9]

  def get_river(self):
    return self.deck[5:10]

