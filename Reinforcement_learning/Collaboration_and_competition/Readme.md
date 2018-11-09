
## The Environment
For this project, we will work with the Tennis environment provided by [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments 
for training intelligent agents.

In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. 
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 
(over 100 consecutive episodes, after taking the maximum over both agents). 
Specifically after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

If you are on a windows machine you won't need to download Unity, but you should make sure to place the folder Tennis_Windows_x86_64
inside the collaboration_and_competition folder.

## Dependencies
To set up your python environment to run the code in this repository, follow the instructions below.

Create (and activate) a new environment with Python 3.6.

#### Linux or Mac:
conda create --name name_of_env python=3.6
source activate name_of_env
#### Windows:
conda create --name name_of_env python=3.6 
activate name_of_env

Clone this repository (if you haven't already!) - you can use git clone - , and navigate to the python/ folder. 
Then, install several dependencies.

*cd python*
*pip install .*

##### Create an IPython kernel for the name_of_env environment 
*python -m ipykernel install --user --name name_of_env --display-name "name_of_env"*

Before running code in a notebook, change the kernel to match the name_of_env environment by using the drop-down Kernel menu.

After you have followed the instructions above, open Tennis.ipynb and follow the instructions to train multiple agents
to play tennis!

## Resources 
[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)


