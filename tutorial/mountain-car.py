import gym
import numpy as np


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000
SHOW_EVERY = 100
STATS_EVERY = 100
DISCRETE_OS_SIZE = [20, 20]

env = gym.make('MountainCar-v0')

env.reset()

high = env.observation_space.high
low = env.observation_space.low

q_table = np.random.uniform(low=-2, high=0, size=(20, 20, 3))

def get_discrete_state(state):
    return tuple(((state - low) / (high - low) * 20).astype(np.int32))

for episode in range(EPISODES):
    print(episode)
    observation = env.reset()
    discrete_state = get_discrete_state(observation)
    done = False
    
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        
        new_observation, reward, done, info = env.step(action)
        new_discrete = get_discrete_state(new_observation)
        
        print(q_table[new_discrete])
        print(np.max(q_table[new_discrete]))
        
        if not done:
            max_future_q = np.max(q_table[new_discrete])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        
        elif(new_observation[0] >= env.goal_position):
            q_table[discrete_state + (action, )] = 0
            print(f'We made it on episode {episode}')
        
        discrete_state = new_discrete
        if render:
            env.render()

env.close()