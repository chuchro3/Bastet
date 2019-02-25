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
import poker_engine
from poker_constants import *


from players.deep_player import DeepPlayer
from players.console_player import ConsolePlayer
from players.random_player import RandomPlayer
from players.fish_player import FishPlayer
from pypokerengine.api.game import setup_config, start_poker

from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck


class ActionSpace(object):
    #valid_actions = ['fold', 'call', 'check', 'raise_half', 'raise_pot', 'raise_2pot', 'all_in']
    valid_actions = ACTIONS
    n = len(valid_actions)
    def __init__(self):
        pass

    def sample(self):
        return self.valid_actions[np.random.randint(0, self.n)]



class PokerEnv(object):
    def __init__(self):
        #4 states
        self.rewards = [0.0] * 10
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        self.action_space = ActionSpace.n
        self.poker = poker_enginer.Poker()

    def reset(self):
        self.cur_state = 0
        self.num_iters = 0
        self.poker = Poker()
        return self.observation_space.states[self.cur_state]
        

    def step(self, action):
        assert(0 <= action <= ActionSpace.n)
        self.num_iters += 1
        assert action in ACTIONS
        poker_action, bet_size = action
        self.poker.make_action(action, bet_size)
        
        rewards = self.reward[0]
        # other player, in this case fish, makes another action
        # if game hasn't ended.
        self.poker.make_action(CALL)

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
