import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    """Entry point into the program.
    """
    env_name = 'CartPole-v1'
    render_mode = 'human'
    env = gym.make(env_name, render_mode=render_mode)

    n_episodes = 5

    for i in range(n_episodes):
        terminated = False
        state = env.reset()
        score = 0

        while terminated == False:
            env.render()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward

        print('Episode {0}: {1}'.format(i, score))

    print('Observation space: {}'.format(env.observation_space))
    print('Action space: {}'.format(env.action_space))
    env.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        raise ex

