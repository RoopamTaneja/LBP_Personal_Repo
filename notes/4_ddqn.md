# Double Deep Q Network

**Double Q-learning** : 

![](img/image-15.jpg)

**DDQN** :

Double Q-learning when combined with deep learning led to Double DQN.

Although not fully decoupled, the target network in the DQN architecture provides
a natural candidate for the second value function, without
having to introduce additional networks. Thus we choose to evaluate the greedy policy according to the Q network, but using the target network to estimate its value.

The observed values are found as earlier ie Q value of current state and action using Q network.

Target value becomes : 

R + gamma * (Q value using *target network* at (next state, a*)) 

where a* = action which gives max Q value using *Q network* for next state across all actions.

Loss function is MSE and target network is updated periodically or using Polyak averaging.

Since DDQN solved overestimation issue, it outperforms DQN generally.