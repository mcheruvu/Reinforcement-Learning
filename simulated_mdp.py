import numpy as np

class SimulatedMDP(object):
    def __init__(self, env, num_episodes=2000):
        """Creating a MDP from simulated experience in an environment.
        
        :param env: environment class which the agent will interact with.
        :param num_episodes: Integer number of episodes to interact.
        """
        
        # OpenAI Gym case.
        try:
            self.n = env.observation_space.n
        # Grid World case.
        except:
            self.n = env.n
        self.states = range(self.n)
        
        # OpenAI Gym case.
        try:
            self.m = env.action_space.n
        # Grid World case.
        except:
            self.m = env.m
        self.actions = range(self.m)

        self.P = np.zeros((self.n, self.m, self.n))
        self.R = np.zeros((self.n, self.m, self.n))

        counts, reward_min = self.simulate_environment(env, num_episodes)
        self.get_learned_model(counts, reward_min)


    def simulate_environment(self, env, num_episodes):
        """Simulate the agent in the environment to get sample transitions.
        
        :param env: environment class which the agent will interact with.
        :param num_episodes: Integer number of episodes to interact.
        :return counts: Numpy array containing counts of transitions for a (s, a, s'). 
        :return reward_min: Float smallest reward encountered in a transition.
        """

        counts = np.zeros((self.n, self.m, self.n))
        reward_min = 100000000
        
        for episode in range(num_episodes):
            
            s = env.reset()
            done = False
            
            while not done:
                # OpenAI Gym case.
                try:
                    a = env.action_space.sample()
                # Grid World Case.
                except:
                    a = env.sample()
                s_new, reward, done, info = env.step(a)
                
                reward_min = min(reward_min, reward)
                
                counts[s, a, s_new] += 1.
                self.R[s, a, s_new] += reward
                
                s = s_new

        return counts, reward_min
        

    def get_learned_model(self, counts, reward_min):
        """Get the learned probability and reward distributions from sampled transitions.
        
        :return counts: Numpy array containing counts of transitions for a (s, a, s'). 
        :return reward_min: Float smallest reward encountered in a transition.
        """

        for s in self.states:
            for a in self.actions:
                # If a state and action was never taken set to uniform probability.
                if counts[s, a].sum() == 0:
                    self.P[s, a] = 1./self.n
                    self.R[s, a] = 0.
                else:
                    # In case of 0/0, this flag will change average to nan.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        self.R[s, a] = self.R[s, a]/counts[s, a]

                    self.P[s, a] = counts[s, a]/counts[s, a].sum()

        # Converting nan value to 0.
        self.R = np.nan_to_num(self.R)

        # Converting learned reward for all transitions not visited to minimum reward.
        self.R[np.where(counts == 0)[0], np.where(counts == 0)[1], np.where(counts == 0)[2]] = reward_min

        self.check_valid_dist()
        
        
    def check_valid_dist(self):
        """Checking the probability distribution sums to 1 for each state, action pair."""

        for s in self.states:
            for a in self.actions:
                assert abs(sum(self.P[s, a, :]) - 1) < 1e-3, 'Transitions do not sum to 1'