import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()

# pip install gym==0.17.3
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.95 
EPISODES = 2000

SHOW_STATS = 500

DESCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
DESCRETE_OBS_WIN_SIZE = (env.observation_space.high - env.observation_space.low)/DESCRETE_OBS_SIZE

print(env.observation_space.low)


print(DESCRETE_OBS_SIZE)
print(DESCRETE_OBS_WIN_SIZE)

epsillon = 0.5
START_EPSILLON_DECAYING = 1
END_EPSILLON_DECAYING = EPISODES//2

epsilon_decay_value = epsillon/(END_EPSILLON_DECAYING - START_EPSILLON_DECAYING)


q_table = np.random.uniform(low=-2,high=0,size=(DESCRETE_OBS_SIZE + [env.action_space.n]))

ep_rewards = []
agg_ep_reward = {'ep':[],'avg':[],'min':[],'max':[]}

def get_discrete_state(state):
    ds = (state - env.observation_space.low) / DESCRETE_OBS_WIN_SIZE
    return tuple(ds.astype(np.int64))

print(env.goal_position)

for ep in range(EPISODES):
    episode_reward = 0
    if ep % SHOW_STATS == 0:
        render = True
        print(ep)
    else : 
        render = False
    print(ep)
    ds = get_discrete_state(env.reset())
    print(ds)
    done = False
    while not done:

        if np.random.random() > epsillon:
            action = np.argmax(q_table[ds])
        else : 
            action = np.random.randint(0,env.action_space.n)
        action = np.argmax(q_table[ds])
        new_state,reward,done,info = env.step(action)
        episode_reward += reward
        new_ds_state = get_discrete_state(new_state)

        if ep % SHOW_STATS == 0:
           env.render()
        
        if  not done:
            max_f_Q = np.max(q_table[new_ds_state])
            current_value = q_table[ds + (action, )]
            new_Q = (1-LEARNING_RATE) * current_value + LEARNING_RATE * (reward + DISCOUNT_RATE + max_f_Q)
            q_table[ds + (action,)] = new_Q
        elif new_state[0] >= env.goal_position:
            q_table[ds + (action, )] = 0
        
        ds = new_ds_state
    if END_EPSILLON_DECAYING >= ep >= START_EPSILLON_DECAYING:
        epsillon -= epsilon_decay_value
    ep_rewards.append(episode_reward)
    if not ep % SHOW_STATS:
        average_reward = sum(ep_rewards[-SHOW_STATS:])/len(ep_rewards[-SHOW_STATS:])
        agg_ep_reward['ep'].append(ep)
        agg_ep_reward['avg'].append(average_reward)
        agg_ep_reward['min'].append(min(ep_rewards[-SHOW_STATS:]))
        agg_ep_reward['max'].append(max(ep_rewards[-SHOW_STATS:]))

        print(f"Episode : {ep} average : {average_reward} min {min(ep_rewards[-SHOW_STATS:])} max {max(ep_rewards[-SHOW_STATS:])}")






env.close()

plt.plot(agg_ep_reward['ep'],agg_ep_reward['avg'],label="avg")
plt.plot(agg_ep_reward['ep'],agg_ep_reward['min'],label="min")
plt.plot(agg_ep_reward['ep'],agg_ep_reward['max'],label="max")
plt.legend(loc = 4)
plt.show()