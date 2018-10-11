# The Environment
For this project, we will work with the [Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Unity ML-Agents
Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments
for training intelligent agents.

Through out this implementation we will use Unity's rich environments to design, train, and evaluate deep reinforcement learning 
algorithms. You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).


There are two separate versions of the Unity environment for this project:

*The first version contains a single agent*
*The second version contains 20 identical agents, each with its own copy of the environment*

We will sove option 2:
The barrier for solving the second version of the environment take into account the presence of many agents. In particular, our agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).


## Step 2: Download the Unity Environment
For this project, if you have a Windows machine, you will not need to install Unity - this is because you will find the environment in the *Reacher_Windows_64* folder. 

## Step 3: Follow the implementation in the continuous_control.ipynb

## Resources 
[DDPG] (https://arxiv.org/abs/1509.02971)

