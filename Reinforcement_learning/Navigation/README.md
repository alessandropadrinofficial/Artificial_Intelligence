
We will use Deep Q-Networks to train an agent inside a [Unity-ML](https://github.com/Unity-Technologies/ml-agents) environment, an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

## Environment
The environment is a open 3D space that the agent will need to navigate. The goal is to collect as many good (yellow) bananas as possible while avoiding bad (blue) ones. 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0. Move forward.

1. Move backward.

2. Turn left.

3. Turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Setup

The notebook requires Python 3.6 along with **Jupyter**, **PyTorch**, **Numpy**, **Matplotlib**, and the **Unity ML-Agents Toolkit**.

Assuming you have the Anaconda package manager installed, a suitable environment (here named navigation) can be created by running the following command:

*conda create -n Navigation python=3.6 jupyter pytorch matplotlib*

Activate the environment and download the Unity ML-Agents Toolkit. Assuming the toolkit has been downloaded to the folder ml-agents, run the following command:

*pip install ml-agents/python*

Now, start a jupyter server in the DRLND-navigation repository's base directory:

*jupyter notebook*

Open the Navigation.ipynb notebook and follow the instructions.

## Resources
[Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)



