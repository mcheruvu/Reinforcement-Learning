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

