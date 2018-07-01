import numpy as np
import itertools
import sys
import os
import string
import matplotlib.pyplot as plt
from matplotlib.cbook import MatplotlibDeprecationWarning
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib as mpl
from pylab import rc
#from itertools import izip
import seaborn as sns
sns.reset_orig()
import warnings

from rl_base import RLBase
from rl_model_free_base import ModelFreeRLBase

class ModelFreeRL(ModelFreeRLBase):
    def __init__(self, n, m, states, actions, gamma=1, alpha=.618, alpha_decay=True, 
                 alpha_decay_param=.001, epsilon=.2, epsilon_decay=True, epsilon_decay_param=.01, 
                 tau=100, tau_decay=True, tau_decay_param=.01, policy_strategy='e-greedy', 
                 horizon=1000, num_episodes=2000):

        super(ModelFreeRL, self).__init__(n, m, states, actions, gamma, alpha, 
                                          alpha_decay, alpha_decay_param,
                                          epsilon, epsilon_decay, epsilon_decay_param, 
                                          tau, tau_decay, tau_decay_param, policy_strategy, 
                                          horizon, num_episodes)

    
    def one_step_temporal_difference(self, env):
        """Finding the value function for a policy using one step temporal difference.

        A policy should already be defined for the class before running this function.
        
        :param env: environment class which the algorithm will attempt to learn.
        """

        self.initialize()

        for self.episode in range(self.num_episodes):

            done = False
            episode_reward = 0
            s = env.reset()
            iteration = 0

            while done != True and iteration < self.horizon:
                a = self.policy[s]
                s_new, reward, done, info = env.step(a)

                self.reward_min = min(self.reward_min, reward)
                episode_reward += reward

                self.visited_states[s, a, s_new] += 1.
                self.experienced_rewards[s, a, s_new] += reward

                self.alpha = self.set_alpha(s)

                # One step temporal difference equation.
                self.v[s] += self.alpha*(reward + self.gamma*self.v[s_new] - self.v[s])

                s = s_new
                iteration += 1

            self.episode_rewards.append(episode_reward)

        self.get_learned_model()


    def sarsa(self, env):
        """Finding the q function using the on policy TD method SARSA.

        :param env: environment class which the algorithm will attempt to learn.
        """

        self.initialize()

        for self.episode in range(self.num_episodes):

            done = False
            episode_reward = 0
            s = env.reset()
            a = self.choose_action(s)
            iteration = 0

            while done != True and iteration < self.horizon:
                s_new, reward, done, info = env.step(a)

                self.reward_min = min(self.reward_min, reward)
                episode_reward += reward

                a_new = self.choose_action(s_new)

                self.visited_states[s, a, s_new] += 1.
                self.experienced_rewards[s, a, s_new] += reward

                alpha = self.set_alpha(s)
                
                # On policy temporal difference update equation.
                self.q[s, a] += alpha*(reward + self.gamma*self.q[s_new, a_new] - self.q[s, a])

                s = s_new
                a = a_new
                iteration += 1

            self.episode_rewards.append(episode_reward)

        self.get_learned_model()
        self.v = self.q.max(axis=1)
        self.policy = super(ModelFreeRL, self).random_policy(self.q)


    def q_learning(self, env):
        """Finding the q function using the off policy TD method q-learning.

        :param env: environment class which the algorithm will attempt to learn.
        """

        self.initialize()

        for self.episode in range(self.num_episodes):

            done = False
            episode_reward = 0
            s = env.reset()
            iteration = 0

            while done != True and iteration < self.horizon:
                a = self.choose_action(s)
                s_new, reward, done, info = env.step(a)

                self.reward_min = min(self.reward_min, reward)
                episode_reward += reward

                self.visited_states[s, a, s_new] += 1.
                self.experienced_rewards[s, a, s_new] += reward

                alpha = self.set_alpha(s)

                # Off policy temporal difference update equation.
                self.q[s, a] += alpha*(reward + self.gamma*self.q[s_new].max() - self.q[s, a])

                s = s_new
                iteration += 1

            self.episode_rewards.append(episode_reward)

        self.get_learned_model()
        self.v = self.q.max(axis=1)
        self.policy = super(ModelFreeRL, self).random_policy(self.q)

