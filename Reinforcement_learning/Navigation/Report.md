The learning algorithm used is vanilla Deep Q Learning as described in original paper. 
As an input the vector of state is used instead of an image so convolutional neural nework is replaced with deep neural network. 
The deep neural network has following layers:

Fully connected layer - input: 37 (state size) output: 64
Fully connected layer - input: 64 output 64
Fully connected layer - input: 64 output: (action size)

### Parameters

Maximum steps per episode: 1000
Starting epsilion: 1.0
Ending epsilion: 0.01
Epsilion decay rate: 0.999

replay buffer size: int(5e6)  
minibatch size: 16     
discount factor: 0.99      
tau = 0.002   (for soft update of target parameters)
learning rate = 5e-4               
The network is updated after every 1 episode

### Performance
The model was trained over 1000 episodes. 
The trend of the average scores is illustrated in the plot below:

![alt text](images/reward.png) 

### Ideas for Future Work

Further hyperparameters optimizations could lead to better performances. Moreover, possible extensions and improvements of the vanilla Deep Q Network are:

- [Double Deep Q Networks](https://arxiv.org/pdf/1509.06461.pdf)
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
- [Dueling Deep Q Networks](https://arxiv.org/pdf/1511.06581.pdf)
- [Distributional DQN](https://arxiv.org/pdf/1707.06887.pdf)
- [Noisy DQN](https://arxiv.org/pdf/1706.10295.pdf)
- [RAINBOW Paper](https://arxiv.org/pdf/1710.02298.pdf)
