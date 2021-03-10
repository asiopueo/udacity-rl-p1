from unityagents import UnityEnvironment
from collections import deque, namedtuple
import numpy as np
import time
import torch

#################################
#  Initialization:
#################################

# Define named tuple 'Experience'; you can use a dictionary alternatively
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64", no_graphics=False)
#env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")
# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Initialize the agent:
#from dqn_agent import DQNAgent
#agent = DQNAgent(buffer_size=10000, batch_size=64, action_size=4, gamma=0.98)

# Alternatively, try out the Double-Q-Learning Agent:
#from ddqn_agent import DDQNAgent
#agent = DDQNAgent(buffer_size=10000, batch_size=64, action_size=4, gamma=0.99, algo_type='GRADIENT_TAPE')
#agent = DDQNAgent(buffer_size=10000, batch_size=64, action_size=4, gamma=0.99, algo_type='COMPILE_FIT')


# Agent based on PyTorch:
from agent_torch import TorchAgent
agent = TorchAgent(buffer_size=100000, batch_size=64, action_size=4, gamma=0.99)


# Initial values:
score_list = []   # Score is NOT the discounted reward but the final 'Banana Score' of the game
score_trailing_list = deque(maxlen=100)
episode = 0


####################################
#  Main learning loop:
####################################


#agent.load_weights("./checkpoints")

N_episodes = 500
eps = 1.
eps_end = 0.02
eps_decay = 0.995
learning_period = 4

while episode in range(N_episodes):
    ticks = 0
    score = 0

    env_info = env.reset(train_mode=True)[brain_name]  # Reset the environment
    state = env_info.vector_observations[0]            # Get the current state
    
    start = time.time()
    while True:
        # Select action according to policy:
        action = agent.action(state, epsilon=eps)

        # Take action and record the reward and the successive state
        env_info = env.step(action)[brain_name]

        reward = env_info.rewards[0]
        next_state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        
        # Add experience to the agent's replay buffer:
        exp = Experience(state, action, reward, next_state, done)
        agent.replay_buffer.insert_into_buffer( exp )

        if ticks % learning_period == 0:
            agent.learn()

        score += reward
        state = next_state

        if done is True:
            break

        ticks += 1

    end = time.time()

    score_list.append(score)
    score_trailing_list.append(score)

    score_avg = np.mean(score_list)
    score_trailing_avg = np.mean(score_trailing_list)

    print("***********************************************")
    print("Score of episode {}: {}".format(episode, score))
    print("Avg. score: {:.2f}".format(score_avg))
    print("Trailing avg. score: {:.2f}".format(score_trailing_avg))
    print("Greedy epsilon used: {:.2f}".format(eps))
    print("Time consumed: {:.2f} s".format(end-start))
    print("***********************************************")

    eps = max(eps*eps_decay, eps_end)
    episode += 1

    agent.save_weights("./checkpoints")


env.close()



