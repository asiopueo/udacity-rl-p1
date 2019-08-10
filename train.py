from unityagents import UnityEnvironment
import numpy as np

# Initialization of Unity environment
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


from Agent import Agent

# Initialize the agent:
agent = Agent()

# Reset the environment
env_info = env.reset(train_mode=False)[brain_name]



# Initial values:
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
time = 0
action = 0  # Initial action: Move forward


while True:
    action = agent.action(state)
    print('Action taken: ', action)

    env_info = env.step(action)[brain_name]
    reward = env_info.rewards[0]
    next_state = env_info.vector_observations[0]
    done = env_info.local_done[0]

    score += reward
    state = next_state

    if time%50 == 0:
        print("Time: {}".format(time))

    time += 1

    if done:
        break



print("Score: {}".format(score))


env.close()
