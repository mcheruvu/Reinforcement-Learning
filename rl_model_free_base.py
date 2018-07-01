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

class ModelFreeRLBase(RLBase):
    def __init__(self, n, m, states, actions, gamma, alpha, alpha_decay, 
                 alpha_decay_param, epsilon, epsilon_decay, 
                 epsilon_decay_param, tau, tau_decay, tau_decay_param, 
                 policy_strategy, horizon, num_episodes):
        """Initialize model free reinforcement learning parameters.
        
        :param n: Integer number of discrete states.
        :param m: Integer number of discrete actions.
        :param states: List of all states.
        :param actions: List of all actions.
        :param gamma: Float discounting factor for rewards in (0,1].
        :param alpha: Float step size parameter for TD step. Typically in (0,1].
        :param alpha_decay: Bool indicating whether to use decay of step size.
        :param alpha_decay_param: Float param for decay given by alpha*e^(-alpha_decay_param * s_t)
        :param epsilon: Float value (0, 1) prob of taking random action vs. taking greedy action.
        :param epsilon_decay: Bool indicating whether to use decay of epsilon over episodes.
        :param epsilon_decay_param: Float param for decay given by epsilon*e^(-epsilon_decay_param * episode)
        :param tau: Float value for temp. param in the softmax, tau -> 0 = greedy, tau -> infinity = random.
        :param tau_decay: Bool indicating whether to use decay of tau over episodes.
        :param tau_decay_param: Float param for decay given by tau*e^(-tau_decay_param * episode)
        :param policy_strategy: String in {softmax, e-greedy, greedy}, exploration vs exploitation strategy.  
        :param horizon: Integer maximum number of steps to run an episode for.
        :param num_episodes: Integer number of episodes to run learning for.
        """

        self.n = n
        self.m = m
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_decay_param = alpha_decay_param
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_param = epsilon_decay_param
        self.tau = tau
        self.tau_decay = tau_decay
        self.tau_decay_param = tau_decay_param
        self.policy_strategy = policy_strategy
        self.horizon = horizon
        self.num_episodes = num_episodes


    def initialize(self):
        """Initialize parameters for learning algorithms."""

        # Q-value function - values of taking action a in state s and acting optimally after.
        self.q = np.zeros((self.n, self.m))

        # Value function - values of state acting optimally from the state.
        self.v = np.zeros(self.n)

        # Counts of state, action, state transition.
        self.visited_states = np.zeros((self.n, self.m, self.n))

        # Cumulative rewards for each state, action, state transition.
        self.experienced_rewards = np.zeros((self.n, self.m, self.n))

        # List tracking rewards of learning algorithm over episodes.
        self.episode_rewards = []

        # List tracking epsilon choices over episodes.
        self.epsilon_choices = []

        # List tracking tau choices over episodes.
        self.tau_choices = []

        # List of lists tracking step size over states (outer) and episodes (inner). 
        self.alpha_choices = [[] for s in self.states]

        """Default min reward which will be changed, so when learning the model
        from experience states that are not visited can be set to minimum reward."""
        self.reward_min = 1000000000
        
        self.total_reward = 0;
        self.avg_reward = 0;


    def choose_action(self, s):
        """Choose action for a TD algorithm that is updating using q values.

        The policy strategy for choosing an action is either chosen using a
        softmax strategy, epsilon greedy strategy, greedy strategy, or a random strategy.

        :param s: Integer index of the current state index the agent is in.

        :return a: Integer index of the chosen index for the action to take.
        """

        if self.policy_strategy == 'softmax':
            a = self.softmax(s)
        elif self.policy_strategy == 'e-greedy':
            a = self.epsilon_greedy(s)
        elif self.policy_strategy == 'greedy':
            a = super(ModelFreeRLBase, self).random_policy(self.q[s])
        else:
            a = np.random.choice(self.actions)

        return a


    def epsilon_greedy(self, s):
        """Epsilon greedy exploration-exploitation strategy.

        This policy strategy selects the current best action with probability
        of 1 - epsilon, and a random action with probability epsilon.
        
        :param s: Integer index of the current state index the agent is in.

        :return a: Integer index of the chosen index for the agent to take.
        """

        if self.epsilon_decay:
            epsilon = self.epsilon * np.exp(-self.epsilon_decay_param * self.episode)
        else:
            epsilon = self.epsilon

        if len(self.epsilon_choices) <= self.episode:
            self.epsilon_choices.append(epsilon)

        if not np.random.binomial(1, epsilon):
            a = super(ModelFreeRLBase, self).random_policy(self.q[s])
        else:
            a = np.random.choice(self.actions)

        return a


    def softmax(self, s):
        """Softmax exploration-exploitation strategy.

        This policy strategy uses a boltzman distribution with a temperature 
        parameter tau, to assign the probabilities of choosing an action based
        off of the current q value of the state and action.

        :param s: Integer index of the current state index the agent is in.

        :return a: Integer index of the chosen index for the agent to take.
        """

        if self.tau_decay:
            # Capping the minimum value of tau to prevent overflow issues.
            tau = max(self.tau * np.exp(-self.tau_decay_param * self.episode), .1)
        else:
            tau = self.tau

        if len(self.tau_choices) <= self.episode:
            self.tau_choices.append(tau)

        exp = lambda s, a: np.exp(self.q[s, a]/tau)
        values = []
        probs = []

        for a in self.actions:
            # Catching overflow and returning greedy action if it occurs.
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    value = exp(s, a)
                except RuntimeWarning:
                    return super(ModelFreeRLBase, self).random_policy(self.q[s])

            values.append(value) 
        
        total = sum(values)
        probs = [val/total for val in values]

        try:
            sample = np.random.multinomial(1, probs).tolist()
            a = sample.index(1)
        except:
            # Return greedy action if there is overflow issues.
            a = super(ModelFreeRLBase, self).random_policy(self.q[s])

        return a


    def set_alpha(self, s):
        """Selecting step size parameter for temporal difference methods.

        :param s: Integer index of the current state the agent is in.
        
        :return alpha: Step size parameter for temporal difference error.
        """

        if self.alpha_decay:
            alpha = self.alpha * np.exp(-self.alpha_decay_param * self.visited_states[s].sum())
        else:
            alpha = self.alpha

        self.alpha_choices[s].append(alpha)

        return alpha


    def get_learned_model(self):
        """Get the learned probability and reward distributions from sampled transitions."""

        self.P = np.zeros((self.n, self.m, self.n))
        self.R = np.zeros((self.n, self.m, self.n))

        for s in self.states:
            for a in self.actions:
                # If a state and action was never taken set to uniform probability.
                if self.visited_states[s, a].sum() == 0:
                    self.P[s, a] = 1./self.n
                    self.R[s, a] = 0.
                else:
                    # In case of 0/0, this flag will change average to nan.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        self.R[s, a] = self.experienced_rewards[s, a]/self.visited_states[s, a]

                    self.P[s, a] = self.visited_states[s, a]/self.visited_states[s, a].sum()

        # Converting nan value to 0.
        self.R = np.nan_to_num(self.R)

        # Converting learned reward for all transitions not visited to minimum reward.
        self.R[np.where(self.visited_states == 0)[0], np.where(self.visited_states == 0)[1], 
                        np.where(self.visited_states == 0)[2]] = self.reward_min

        self.check_valid_dist()


    def check_valid_dist(self):
        """Checking the probability distribution sums to 1 for each state, action pair."""

        for s in self.states:
            for a in self.actions:
                assert abs(sum(self.P[s, a, :]) - 1) < 1e-3, 'Transitions do not sum to 1'


    def plot_alpha_parameters(self, s, title='State Transition over Episodes', fig_path=None, 
                              fig_name=None, save_fig=True):
        """Plotting the step size choices for a state over episodes.
        
        :param s: Integer state index to plot the step size choices for.
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.plot(self.alpha_choices[s], color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('State Visit Number', fontsize=20)
        plt.ylabel(r'$\alpha_s$', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.alpha_choices[s])])

        plt.tight_layout()

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/images'

            # Default figure name.
            if fig_name is None:
                #title = title.translate(string.punctuation)
                #fig_name = '_'.join(title.split()) + '.png'
                fig_name = title.replace(' ', '-').lower() + '.png'

            plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

        sns.reset_orig()

        plt.show()


    def plot_epsilon_parameters(self, title='Epsilon over Episodes', fig_path=None, 
                                fig_name=None, save_fig=True):
        """Plotting the e-greedy parameter over episodes.
        
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.plot(self.epsilon_choices, color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('Episodes', fontsize=20)
        plt.ylabel(r'$\epsilon$', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.epsilon_choices)])

        plt.tight_layout()

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/images'

            # Default figure name.
            if fig_name is None:
                #title = title.translate( string.punctuation)
                #fig_name = '_'.join(title.split()) + '.png'
                fig_name = title.replace(' ', '-').lower() + '.png'
            
            plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

        sns.reset_orig()

        plt.show()


    def plot_tau_parameters(self, title='Tau Parameters', fig_path=None, 
                            fig_name=None, save_fig=True):
        """Plotting softmax parameter tau over episodes.
        
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.plot(self.tau_choices, color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('Episodes', fontsize=20)
        plt.ylabel(r'$\tau$', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.tau_choices)])

        plt.tight_layout()

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/images'
                
            # Default figure name.
            if fig_name is None:
                #title = title.translate(string.punctuation)
                #fig_name = '_'.join(title.split()) + '.png'
                fig_name = title.replace(' ', '-').lower() + '.png'
            
            plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

        sns.reset_orig()

        plt.show()

