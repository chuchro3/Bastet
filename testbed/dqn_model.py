#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:01:36 2019

@author: javenxu
"""
import tensorflow as tf
import sys
sys.path.append('../')
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear

from configs.q3_nature import config

import numpy as np
import pypokerengine
import pprint
from players.deep_player import DeepPlayer
from players.console_player import ConsolePlayer
from players.random_player import RandomPlayer
from players.fish_player import FishPlayer
from pypokerengine.api.game import setup_config, start_poker

from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck


INITIAL_STACK = 1000 
NUMBER_PLAYERS = 2
MAX_ROUND = 1
SMALL_BLIND_AMOUNT = 10
ANTE_AMOUNT = 0

CALL = 100
FOLD = 200
CHECK = 300
RAISE = 400
ALL_IN = 500
ACTIONS = [CALL, FOLD, CHECK, RAISE, ALL_IN]

class ActionSpace(object):
    #valid_actions = ['fold', 'call', 'check', 'raise_half', 'raise_pot', 'raise_2pot', 'all_in']
    valid_actions = [CALL, FOLD, CHECK, RAISE, ALL_IN]
    n = len(valid_actions)
    def __init__(self):
        pass

    def sample(self):
        return self.valid_actions[np.random.randint(0, self.n)]

class Poker():
    def __init__(self, dealer=0):
        self.deck = random.shuffle(range(52))
        self.p1_stack = 1000
        self.p2_stack = 1000
        self.p1_turn = True
        # 0:p1, 1:p2
        self.dealer = dealer
        # 0: pre-flop, 1: flop, 2: turn, 3: river, 4: done
        self.stage = 0
        self.pot = 0
        self.p1_cards = self.deck[0:2]
        self.p2_cards = self.deck[2:4]
        self.common_cards = None
        self.states = [self.p1_turn, self.pot, self.p1_cards, self.common_cards]
        
    def bet(self, player, bet):
        if bet is FOLD:
            self.stage = 4
        
            self.

    def next_turn(self):
        if self.stage == 0:
            self.common_cards = self.deck[5:8]
        elif self.stage == 1:
            self.common_cards = self.deck[5:9]
        elif self.stage == 2:
            self.common_cards = self.deck[5:10]
        
        self.stage += 1
        return


class PokerEnv(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    Modified 
    """
    def __init__(self):
        #4 states
        self.rewards = [0.0] * 10
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        self.action_space = ActionSpace.n
        self.poker = Poker(shape)
        # poker stuff
        self.poker = Poker()

    def reset(self):
        self.cur_state = 0
        self.num_iters = 0
        self.poker = Poker()
        return self.observation_space.states[self.cur_state]
        

    def step(self, action):
        assert(0 <= action <= ActionSpace.n)
        self.num_iters += 1
        assert action in ACTIONS




        return self.observation_space.states[self.cur_state], reward, self.num_iters >= 5, {'ale.lives':0}


    def render(self):
        print(self.cur_state)

class NatureQN(Linear):
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # num_actions = self.env.action_space.n

#        with tf.variable_scope(scope, reuse=reuse):
#            out = tf.layers.conv2d(state, filters=32, kernel_size=(8, 8),strides=4, activation=tf.nn.relu,
#                                   padding="same")
#            out = tf.layers.conv2d(out, filters=64, kernel_size=(4, 4),strides=2, activation=tf.nn.relu,
#                                   padding="same")
#            out = tf.layers.conv2d(out, filters=64, kernel_size=(3, 3),strides=1, activation=tf.nn.relu,
#                                   padding="same")
#            
#            # flatten
#            out = tf.layers.flatten(out)
#            out = tf.layers.dense(out, units=512, activation=tf.nn.relu)
#            out = tf.layers.dense(out, units=num_actions, activation=None)

        ##############################################################
        return [0.0]*4
        # return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
