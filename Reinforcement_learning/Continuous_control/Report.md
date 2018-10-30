I chose to solve this environment with Deep Deterministic Policy Gradient (DDPG) since it works well with continuous action spaces.
It uses an actor-critic model, using value based methods to reduce the variance of policy based methods.

## Model
- Actor is a MLP with two hidden layers of 400 and 300 units and Relu activation. Actions are bounded from a Tanh activation
- Critic is a MLP with two hidden layers of 400 and 300 units and Relu activations. Action signal is fed to the critic network
at the second layer.

## Hyperparameters
Here the chosen hyperparameters for both single and multi agents environment:

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 128        
GAMMA = 0.99            
TAU = 1e-3              
LR_ACTOR = 1e-4          
LR_CRITIC = 3e-4        
WEIGHT_DECAY = 0.0001   

## Tips
Major improvements:
- I found that gradient clipping for critic's parameters only improves the performance in multi-agent training while in the single agent 
environment degraded it.
You can use *torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)* to clip the gradient.
- Parallelization seems to reduce the bias and improve stability during training.
- Using a normal distribution (np.random.randn) when adding randomness to the Ornstein-Uhlenbeck process. 


## Results
The environment was solved in 181 episodes, 
here you can check the reward plot over 100 episodes and the reward plot over the 181 episodes :


![alt text](Images/average_reward_over100_reacher.png) 
![alt text](Images/average_reward_over_all_reacher.png) 


## Ideas for future work

- Add [noise to the policy parameters](https://blog.openai.com/better-exploration-with-parameter-noise/)
- Using [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617)
- Use prioritized experience replay. 
