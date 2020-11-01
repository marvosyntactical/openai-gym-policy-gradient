import gym
# from policy_gradient_layers import PolicyGradient
from pg_keras import PolicyGradient


import matplotlib.pyplot as plt
import numpy as np

import os

task = "Ant-v2"
env = gym.make(task)
env = env.unwrapped
# env = env.env

# Policy gradient has high variance, seed for reproducibility
env.seed(42)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)

RENDER_ENV = False
EPISODES = 500
RENDER_REWARD_MIN = 50

rewards = []

if __name__ == "__main__":

    # Load checkpoint
    load_path = None #"output/weights/CartPole-v0.ckpt"
    save_path = None #"output/weights/CartPole-v0-temp.ckpt"

    PG = PolicyGradient(
        n_x = env.observation_space.shape[0],
        n_y = env.action_space.shape[0],
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=load_path,
        save_path=save_path
    )

    for episode in range(EPISODES):

        observation = env.reset()

        while True:
            if RENDER_ENV and os.environ.get("DISPLAY"):
                env.render()

            # 1. Choose an action based on observation
            action = PG.choose_action(observation)

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)

            # 3. Store transition for training
            PG.store_transition(observation, action, reward)

            if done:
                episode_rewards = PG.episode_rewards

                episode_rewards_sum = sum(episode_rewards)
                rewards.append(episode_rewards_sum)

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)
                print("Max reward so far: ", max_reward_so_far)

                """
                print("TF total params (should not change...):  ", params)
                episode_len = len(episode_rewards)
                print("TF Flops: ", flops*episode_len)
                """

                # 4. Make Gradient Step 
                discounted_episode_rewards_norm = PG.learn()

                # Render env if we get to rewards minimum
                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True

                break

            # Save new observation
            observation = observation_
    
    print("Finished all episodes, bye :*")
    print("Max reward: ", max_reward_so_far)
    
