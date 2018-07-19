Flappy Bird

Produce a Table of RL-based approaches (Policy Iteration, Value Iteration, Q Learning, TD, Monte Carlo) 
where each row is an approach and each column is a dimension that differentiates the approaches such as model free, 
which of the quintuple each uses (State, Action, Transitions, Rewards, etc..), and other dimensions.


Flappybird is a side-scrolling game where the agent must successfully navigate through gaps between pipes. The up arrow causes the bird to accelerate upwards. If the bird makes contact with the ground or pipes, or goes above the top of the screen, the game is over. For each pipe it passes through it gains a positive reward. Each time a terminal state is reached it receives a negative reward.

Determine a good policy for Flappy Birds using any one or more of the following algorithms (aim to get 140 points or more!):

Policy Iteration, Value Iteration, Q Learning, TD, Monte Carlo
You may have to discretize the space of following parameters.

Vertical distance from lower pipe
Horizontal distance from next pair of pipes
Life: Dead or Living

Actions
For each state, there two possible actions
Click
Do Nothing
1.4  Rewards
The reward structure is purely based on the "Life" parameter. One possible such structure could be the following (feel free to explore more):
+1 if Flappy Bird is still alive
-1000 if Flappy Bird is dead
1.5  Flappy Birds Simulator:
Please use the openai gym environment for this project:

https://gym.openai.com/envs/FlappyBird-v0/
1.6  Submission
Part of your your report should be a video of how your agent learns showing progress after: after 10 minutes of training; after 30 minutes; after 5 hours etc.; In addition please submit a notebook (with documented code) and a discussion section.

References:
https://medium.com/@videshsuman/using-reinforcement-learning-techniques-to-build-an-ai-bot-for-the-game-flappy-bird-30e0fd22f990


https://hardikbansal.github.io/FlappyDQNBlog/
https://github.com/chncyhn/flappybird-qlearning-bot

https://github.com/yenchenlin/DeepLearningFlappyBird

https://github.com/aronszanto/Flappy-Bird-Learning/blob/master/FB%20White%20Paper.pdf

https://github.com/rabbitnoname/rlsimple/blob/master/DQN/deep_q_network.py

https://github.com/floodsung/Gym-Flappy-Bird/blob/master/gym_flappy_bird/envs/flappy_bird_env.py

https://github.com/SupaeroDataScience/RLchallenge/blob/master/RandomBird/FlappyAgent.py

https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/docs/user/home.rst

https://nthu-datalab.github.io/ml/labs/16-1_Q-Learning/16-1_Q_Learning.html

Deep Reinforcement Learning nice material:
http://karpathy.github.io/2016/05/31/rl/

Sample Frozen lake environment - Q-learning:
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

diagrams:
https://github.com/mebusy/notes/blob/master/dev_notes/RL_DavidSilver.md

Very good RL OOP algorithms:
https://github.com/fiezt/Reinforcement-Learning/blob/master/code/OpenAIGymExamples.ipynb


Bellman Error:
https://github.com/mpatacchiola/dissecting-reinforcement-learning/blob/master/src/6/multi-armed-bandit/epsilon_greedy_agent_bandit.py


New Ref:

https://dunglai.github.io/2017/09/21/FlappyBirdAI/

https://blog.openai.com/evolution-strategies/

http://blog.aylien.com/flappy-bird-and-evolution-strategies-an-experiment/

using Keras - very good one:
https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
and the code: https://github.com/yanpanlau/Keras-FlappyBird
