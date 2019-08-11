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
state = env_info.vector_observations[0]   # get the current state
score = 0   # Score is NOT the discounted reward but the final 'Banana Score' of the game
time = 0
action = 0  # Initial action: Move forward

#################################
#   Play episodes:
#################################
for i in range(100)
    # Select action according to policy:
    action = agent.action(state)
    print('Action taken: ', action)

    # Take action and record the reward and the successive state
    env_info = env.step(action)[brain_name]
    reward = env_info.rewards[0]
    next_state = env_info.vector_observations[0]
    done = env_info.local_done[0]

    # Add experience to the agent's replay buffer:
    experience = (state,action,reward,next_state,done)
    agent.add( experience )

    # If buffer is suffieicently full, let the agent learn from his experience:
    if agent.memory.length():
        agent.learn()

    score += reward
    state = next_state

    if time%10 == 0:
        pass
        #agent.update_long_net()

    if time%50 == 0:
        print("Time: {}".format(time))

    time += 1

    if done:
        break



print("Score: {}".format(score))


env.close()
