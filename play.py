from unityagents import UnityEnvironment
import numpy as np

#################################
#  Initialization:
#################################
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64", no_graphics=False)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from collections import namedtuple


# Initialize the agent:
#agent = DQNAgent(buffer_size=10000, batch_size=64, action_size=4, gamma=0.98)
# Alternatively, try out the Double-Q-Learning Agent:
agent = DDQNAgent(buffer_size=10000, batch_size=64, action_size=4, gamma=0.98)

# Define named tuple 'Experience'; you can use a dictionary alternatively
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# Initial values:
score_avg = 0   # Score is NOT the discounted reward but the final 'Banana Score' of the game
episode = 0

####################################
#  Step-function and Main Loop:
####################################


#agent.load_weights("./checkpoints")

N_episodes = 500
eps = 1.0
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

while episode in range(N_episodes):
    time = 0
    score = 0

    env_info = env.reset(train_mode=False)[brain_name]  # Reset the environment
    state = env_info.vector_observations[0]             # Get the current state
    
    while True:
        # Select action according to policy:
        action = agent.action(state, epsilon=eps)
        #print("[Episode {}, Time {}] Action taken: {}".format(episode, time, action))

        # Take action and record the reward and the successive state
        try:
            env_info = env.step(action)[brain_name]
        except:
            print("Total score: {}".format(score))
            env.close()

        reward = env_info.rewards[0]
        next_state = env_info.vector_observations[0]
        done = env_info.local_done[0]

        # Add experience to the agent's replay buffer:
        exp = Experience(state, action, reward, next_state, done)
        agent.replay_buffer.insert_into_buffer( exp )

        agent.learn()

        score += reward
        state = next_state

        if done is True:
            break

        time += 1

    eps = max(eps*eps_decay, eps_end)
    score_avg = (score_avg*episode + score) / (episode+1)

    print("***********************************************")
    print("Score of episode {}: {}".format(episode, score))
    print("Avg. score: {}".format(score_avg))
    print("***********************************************")
    episode += 1
    agent.save_weights("./checkpoints")



env.close()



