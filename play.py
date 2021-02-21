from unityagents import UnityEnvironment
import numpy as np

#################################
#  Initialization:
#################################
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

from Agent import Agent
from collections import namedtuple


# Initialize the agent:
agent = Agent(buffer_size=1000, batch_size=30, action_size=4, gamma=0.98)

# Define named tuple 'Experience'; you can use a dictionary alternatively
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# Initial values:
score_avg = 0   # Score is NOT the discounted reward but the final 'Banana Score' of the game
episode = 0

####################################
#  Step-function and Main Loop:
####################################
def step():
    global score, time, state, env_info

    # Select action according to policy:
    action = agent.action(state, epsilon=0.9)
    #action = agent.random_action()
    print("[Episode {}, Time {}] Action taken: {}".format(episode, time, action))

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

    # If buffer is sufficiently full, let the agent learn from his experience:
    if agent.replay_buffer.buffer_usage():
        agent.learn()

    score += reward
    state = next_state

    return done
        


agent.load_weights("./checkpoints")

N_episodes = 100

while episode in range(N_episodes):
    time = 0
    score = 0

    env_info = env.reset(train_mode=False)[brain_name]  # Reset the environment
    state = env_info.vector_observations[0]             # Get the current state
    
    while True:
        done = step()

        if done is True:
            break
        
        #print("[Time: {}] Score {}".format(time, score))

        if time%10 == 0:
            pass
            #print("[Time: {}] Buffer usage: {}".format(time, agent.replay_buffer.buffer_usage()))
        elif time%25 == 0:
            agent.update_target_net()

        time += 1
    
    score_avg = score_avg*episode / (episode+1)

    print("***********************************************")
    print("Score of episode {}: {}".format(episode, score))
    print("Avg. score: {}".format(score_avg))
    print("***********************************************")
    episode += 1
    agent.save_weights("./checkpoints")





