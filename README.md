[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation
This project is about training an agent to navigate and collect bananas in a large, rectangular world.  

![Trained Agent][image1]

There are two types of bananas available for collection: yellow bananas which - upon collection - provide a reward of +1 points, and blue bananas with a negative reward of -1 points. 
The task is episodic and each episode consists of 300 consecutive steps. In order to succeed, the agent needs an average score of +13 points over 100 consecutive episodes.

There are four discrete actions available for the agent at each step:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The state space consists of 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  



## Dependencies and Setup
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

## Structure of the Project
Aside from the Jupyter-Notebook-file `Navigation.ipynb` the project consists of the following support files:

1. `play.py`: Scripting and basic gaming routine
2. `train.py`: Training script for the agent
3. `Agent.py`: Agent class
4. `Network.py`: Contains the definition of the ANN
5. `ReplayBuffer.py`: Replay buffer class

Both `play.py` and `train.py` are not necessary when launching the agent in the Jupyter notebook. However, they are the starting point when controlling the agent from the CLI. Their contents are largely identical with the content in the notebook.

The chosen deep-learning-framework is Keras with *TensorFlow 1.17* backend.


The initial development was largely done from the command line with the tuning of the learning parameters mostly done in the Jupyter-notebook.




## Deep Q-Learning

### Monte-Carlo Learning

$Q_\pi(s,a) = Q_\pi(s,a) + \alpha \left(G_t - Q_\pi(s,a)\right)$

$G_t$ = total reward for the whole episode


### Temporal-Difference Learning (TD-Learning)

Update equation for the Q-value: 

$Q_\pi(s,a) = Q_\pi(s,a) + \alpha\left( r(s,a) + Q_\pi(s',a') - Q_\pi(s,a)\right)$

SARSA-max:

$Q_\pi(s,a) = Q_\pi(s,a) + \alpha \left(r(s,a)+\max_{a'} Q_\pi(s',a') - Q_\pi(s,a\right)$

Update equation for the weights:

$w = w + \alpha\left( r(s,a)+\max_{a'} Q_\pi(s',a',w)-Q_\pi(s,a,w)\right) \nabla_w Q_\pi(s,a,w)$


## Agent Class
The structure of the Agent class is as follows:
```
Agent()
+ action()
+ learn()
+ update_target_net()
```
* The `action()`-method contains the code for the epsilon-greedy-strategy with epsilon being the parameter $0<\epsilon\ll1$
* The `learn()`-method retrieves a batch of memories from the replay buffer and utilizes a gradient policies algorithm to train the agent.
* Finally, `update_target_net()` 


## Replay Buffer Class
```
ReplayBuffer()
+ insert_into_buffer()
+ sample_from_buffer()
+ buffer_usage()
```


## Network
The artificial neural network for the simple problem consists of the following:




## Learning Strategy
The chosen learning strategy is policy gradient.

In practice,

After a predetermined cycle, the target network is updated by the `update_target_net()`-method using the parameter $0<\tau<1$:

$w = w + \Delta w$

Written in code, it is as follows:

    self.target_net.set_weights( tau*self.local_net.get_weights() + (1-tau)*self.target_net.get_weights() )

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">


## Deep Q-Learning

There are two important strategies to tackle the instability issues:

* Experience replay
* Fixed Q-targets
* Training of a target network




## Results
For the results, please refer to `Navigation.ipynb`. There you find the final version of the source code, as well as the resulting graphs.

## Literature





