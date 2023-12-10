import os
import sys
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class CommandArgs():
    """Container for command line arguments.

    Attributes
    ----------
    demo : bool
        Activates environment demonstration and rendering prior to
        the training step.

    train : bool
        Activates policy training and saving to disk.

    eval : bool
        Activates policy evaluation after the training step.
    """
    def __init__(self):
        self.demo = False
        self.train = False
        self.evaluate = False

def parse_command_line_args():
    """Extracts the command line arguments.

    Returns
    -------
    command_args : CommandArgs
        The parsed command line arguments.
    """
    i = 1 # Start at 1 to skip the filepath argument
    command_args = CommandArgs()
    while i < len(sys.argv) - 1:
        if str.lower(sys.argv[i]) == '-train':
            if str.lower(sys.argv[i + 1]) == 'true':
                command_args.train = True
            elif str.lower(sys.argv[i + 1]) == 'false':
                command_args.train = False
            else:
                raise ValueError(
                    'Invalid argument found for -train: {}'.format(
                        sys.argv[i + 1]))
            i += 1
        elif str.lower(sys.argv[i]) in ['-eval']:
            if str.lower(sys.argv[i + 1]) == 'true':
                command_args.evaluate = True
            elif str.lower(sys.argv[i + 1]) == 'false':
                command_args.evaluate = False
            else:
                raise ValueError(
                    'Invalid argument found for -eval: {}'.format(
                        sys.argv[i + 1]))
            i += 1
        elif str.lower(sys.argv[i]) in ['-demo']:
            if str.lower(sys.argv[i + 1]) == 'true':
                command_args.demo = True
            elif str.lower(sys.argv[i + 1]) == 'false':
                command_args.demo = False
            else:
                raise ValueError(
                    'Invalid argument found for -eval: {}'.format(
                        sys.argv[i + 1]))
            i += 1
        else:
            raise ValueError('Invalid argument: {}'.format(sys.argv[i]))
        i += 1
    return command_args

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
    command_args = parse_command_line_args()

    output_dir = 'output/'
    log_dir = output_dir + 'logs/'
    model_dir = output_dir + 'models/'
    make_dir(output_dir)
    make_dir(log_dir)
    make_dir(model_dir)
    ppo_path = model_dir + 'ppo_model_cartpole'

    env_id = 'CartPole-v1'
    n_steps = 50000

    if command_args.demo == True:        
        n_episodes = 5
        env = gym.make(id=env_id, render_mode='human')
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
        env.close()

    if command_args.train == True:
        env_train = gym.make(id=env_id)
        env_train = DummyVecEnv([lambda: env_train])
        model = PPO('MlpPolicy', env_train, verbose=1, tensorboard_log=log_dir)
        model.learn(total_timesteps=n_steps)
        model.save(ppo_path)
        env_train.close()

    if command_args.evaluate == True:
        env_eval = gym.make(id=env_id, render_mode='human')
        model = PPO.load(ppo_path)
        evaluate_policy(model, env_eval, n_eval_episodes=5, render=True)
        env_eval.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        raise ex

