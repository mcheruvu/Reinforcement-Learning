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

class ModelBasedRL(RLBase):
    def __init__(self, gamma=1, max_iter=5000, max_eval=1000):
        """
        :param gamma: Float discounting factor for rewards in (0,1].
        :param max_iter: Integer max number of iterations to run policy evaluation and improvement.
        :param max_eval: Integer max number of evaluations to run on each state or each state and action.
        """

        self.gamma = gamma
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.total_reward = 0;
        self.avg_reward = 0;


    def get_policy(self, mdp):
        """Given the value function find the policy for actions in states.
    
        :param mdp: Markov decision process object containing standard information.
        """
        
        self.action_vals = np.zeros((mdp.n, mdp.m))

        for s in mdp.states:
            self.action_vals[s] = [sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v)) 
                                       for a in mdp.actions]

        self.policy = super(ModelBasedRL, self).random_policy(self.action_vals)

    
    def iterative_policy_evaluation(self, mdp, pi=None):
        """Iterative policy evaluation finds the state value function for a policy.

        :param mdp: Markov decision process object containing standard information.
        :param pi: Probability distribution of actions given states.
        """
        
        # Random policy if a policy is not provided.
        if pi is None:
            mdp.pi = 1/float(mdp.m) * np.ones((mdp.n, mdp.m))
        else:
            mdp.pi = pi
        
        self.v = np.zeros(mdp.n)

        for iteration in range(self.max_iter):
            
            delta = 0

            for s in mdp.states:            
                v_temp = self.v[s].copy()       
                
                # Bellman equation to back up.
                self.v[s] = sum(mdp.pi[s, a] * sum(mdp.P[s, a] 
                                * (mdp.R[s, a] + self.gamma*self.v)) 
                                for a in mdp.actions)

                delta = max(delta, abs(v_temp - self.v[s]))

            # Convergence check.
            if delta < 1e-4:
                break
        
        self.get_policy(mdp)
 

    def policy_iteration(self, mdp):
        """Finds optimal policy and the value function for that policy.
        
        :param mdp: Markov decision process object containing standard information.
        """
        
        self.v = np.zeros(mdp.n)
        self.policy = np.zeros(mdp.n, dtype=int)
        self.action_vals = np.zeros((mdp.n, mdp.m))

        # Policy evaluation followed by policy improvement until convergence.
        for iteration in range(self.max_iter):

            # Policy evaluation.
            for evaluation in range(self.max_eval):
                
                delta = 0

                for s in mdp.states:    
                    v_temp = self.v[s].copy()       
                    a = self.policy[s]

                    # Bellman equation to back up.
                    self.v[s] = sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v)) 

                    delta = max(delta, abs(v_temp - self.v[s]))

                # Convergence check.
                if delta < 1e-4:
                    break

            # Policy improvement.
            stable = True

            for s in mdp.states:
                old_policy = self.policy[s].copy()

                self.action_vals[s] = [sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v)) 
                                           for a in mdp.actions]

                self.policy[s] = super(ModelBasedRL, self).random_policy(self.action_vals[s])

                if self.policy[s] != old_policy and stable:
                    stable = False

            # Policy convergence check.
            if stable:
                break
        
        # Policy probability distribution.
        self.pi = np.zeros((mdp.n, mdp.m))
        self.pi[np.arange(self.pi.shape[0]), self.policy] = 1. 


    def value_iteration(self, mdp):
        """Find the optimal value function and policy with value iteration.
        
        :param mdp: Markov decision process object containing standard information.
        """

        self.v = np.zeros(mdp.n)
        self.policy = np.zeros(mdp.n, dtype=int)

        # Value iteration step which effectively combines evaluation and improvement.
        for evaluation in range(self.max_eval):
            
            delta = 0

            for s in mdp.states:            
                v_temp = self.v[s].copy()       
                
                # Bellman equation to back up.
                self.v[s] = max([sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v)) 
                                     for a in mdp.actions])

                delta = max(delta, abs(v_temp - self.v[s]))

            # Convergence check.
            if delta < 1e-4:
                break

        self.get_policy(mdp)

        # Setting policy probability distribution to the greedy policy.
        self.pi = np.zeros((mdp.n, mdp.m))
        self.pi[np.arange(self.pi.shape[0]), self.policy] = 1. 


    def q_value_iteration(self, mdp):
        """Find the optimal q function using q value iteration.

        :param mdp: Markov decision process object containing standard information.
        """

        # Initializing q function.
        self.q = np.zeros((mdp.n, mdp.m))

        for evaluation in range(self.max_eval):

            delta = 0

            for state_action in itertools.product(mdp.states, mdp.actions):
                s = state_action[0]
                a = state_action[1]

                q_temp = self.q[s, a].copy()

                # Bellman equation to back up.
                self.q[s, a] = sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.q.max(axis=1)))

                delta = max(delta, abs(self.q[s, a] - q_temp))

            # Convergence check.
            if delta < 1e-4:
                break

        self.v = self.q.max(axis=1)
        self.policy = super(ModelBasedRL, self).random_policy(self.q)

        # Setting policy probability distribution to the greedy policy.
        self.pi = np.zeros((mdp.n, mdp.m))
        self.pi[np.arange(self.pi.shape[0]), self.policy] = 1. 


    def test_optimal_q(self, mdp):
        """Testing if the q function is optimal.

        :param mdp: Markov decision process object containing standard information.
        """

        self.error = np.zeros((mdp.n, mdp.m))

        for s, a in itertools.product(mdp.states, mdp.actions):
            self.error[s, a] = sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.q.max(axis=1) 
                                   - self.q[s,a]))

        self.error = abs(self.error)


    def test_optimal_v(self, mdp):
        """Testing if the value function is optimal.

        :param mdp: Markov decision process object containing standard information.
        """

        self.error = np.zeros(mdp.n)

        for s in mdp.states:                
            self.error[s] = max([sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v - self.v[s]))
                                 for a in mdp.actions])

        self.error = abs(self.error)

