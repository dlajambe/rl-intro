import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def make_dir(dir: str):
    """Creates a folder at the provided directory if one does not
    already exist.

    Parameters
    ----------

    dir : str
        The location of the directory that should be created.
    """
    if os.path.isdir(dir) == False:
        os.mkdir(dir)

def main():
    """Entry point into the program.
    """
    output_dir = 'output/'
    log_dir = output_dir + 'logs/'
    model_dir = output_dir + '/models'
    make_dir(output_dir)
    make_dir(log_dir)
    make_dir(model_dir)

    env_id = 'CartPole-v1'
    render_mode = 'human'
    env = gym.make(id=env_id, render_mode=render_mode)

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

