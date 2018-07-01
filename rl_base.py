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
import string
    
class RLBase(object):
    def __init__(self):
        pass

    def simulate_policy(self, env, num_episodes=1000, render_final=True):
        """Interact in the environment using the learned policy.

        This function is used to interact in the environment and get the average
        reward over of number of episodes and show the last episode if specified.
        The trajectories are also stored of the state action pairs in each episode.

        :param env: environment class which the algorithm will attempt to learn.
        :param num_episodes: Integer number of episodes to run in environment for.
        :param render_final: Bool indicator of whether to show the last episode.

        :return t
        """

        # List tracking rewards of learning algorithm over episodes.
        self.episode_rewards = []

        # List of episode trajectories containing state action pairs.
        self.trajectories = []

        self.total_reward = 0

        for episode in range(num_episodes):

            epsiode_trajectory = []

            done = False
            episode_reward = 0
            s = env.reset()

            while done != True:
                # Show the environment to the screen.
                if episode == num_episodes - 1 and render_final:
                    env.render()

                a = self.policy[s]

                epsiode_trajectory.append((s, a))

                s, reward, done, info = env.step(a) 

                episode_reward += reward

            self.trajectories.append(epsiode_trajectory)
            self.episode_rewards.append(episode_reward)
            self.total_reward += episode_reward

        self.avg_reward = self.total_reward/float(num_episodes)
        
        print('\n\nAverage reward over %d episodes is %f\n\n' % (num_episodes, self.avg_reward))
        #return num_episodes, total_reward, avg_reward
        
    def random_policy(self, arr):
        """Helper function to get the argmax of an array breaking ties randomly.
        
        :param arr: 1D or 1D numpy array to find the argmax for.
    
        :return choice or argmax_array: Choice is integer index of array with 
        the max value, argmax_array is array of integer index of max value in each 
        row of the original array.
        """
    
        if len(arr.shape) == 1:
            choice = np.random.choice(np.flatnonzero(arr == arr.max()))
            return choice
        else:
            N = arr.shape[0]
            argmax_array = np.zeros(N)
    
            for i in range(N):
                choice = np.random.choice(np.flatnonzero(arr[i] == arr[i].max()))
                argmax_array[i] = choice
    
        argmax_array = argmax_array.astype(int)
    
        return argmax_array
    
    def return_rmse(self, predictions, targets):
        """Return the Root Mean Square error between two arrays
        @param predictions an array of prediction values
        @param targets an array of target values
        @return the RMSE
        """
        return np.sqrt(((predictions - targets)**2).mean())
    
    def plot_epsiode_returns(self, title='Episode vs. Rewards', fig_path=None, 
                             fig_name=None, save_fig=True):
        """Plotting the reward returns over episodes.
        
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.plot(self.episode_rewards, color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('Episodes', fontsize=20)
        plt.ylabel('Cumulative Rewards', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.episode_rewards)])

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


    def scatter_epsiode_returns(self, title='Episode vs. Rewards', fig_path=None, 
                                fig_name=None, save_fig=True):
        """Scatter plotting the reward returns over episodes.
        
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.scatter(range(len(self.episode_rewards)), self.episode_rewards, color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('Episodes', fontsize=20)
        plt.ylabel('Cumulative Rewards', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.episode_rewards)])

        plt.tight_layout()

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/images'

            #print(fig_path)
            # Default figure name.
            if fig_name is None:
                #title = title.translate(string.punctuation)
                #fig_name = '_'.join(title.split()) + '.png'
                fig_name = title.replace(' ', '-').lower() + '.png'

            
            plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

        sns.reset_orig()

        plt.show()

