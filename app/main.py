import os
import sys
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def parse_command_line_args() -> bool:
    """Extracts the command line arguments.

    Returns
    -------
    train : bool
        Indicates whether the policy should be trained or not.
    """
    i = 1 # Start at 1 to skip the filepath argument
    train = True
    while i < len(sys.argv) - 1:
        if str.lower(sys.argv[i]) == '-train':
            if str.lower(sys.argv[i + 1]) == 'true':
                train = True
            elif str.lower(sys.argv[i + 1]) == 'false':
                train = False
            else:
                raise ValueError(
                    'Invalid argument found for -train: {}'.format(
                        sys.argv[i + 1]))
            i += 1
        else:
            raise ValueError('Invalid argument: {}'.format(sys.argv[i]))
        i += 1
    return train

def make_dir(dir: str) -> None:
    """Creates a folder at the provided directory if one does not
    already exist.

    Parameters
    ----------

    dir : str
        The location of the directory that should be created.
    """
    if os.path.isdir(dir) == False:
        os.mkdir(dir)

def main() -> int:
    """Entry point into the program.

    Returns
    -------
    int
        An integer indicating if the program successfully terminated (1)
    """

    # Step 1 - Define hyperparameters and paths
    train_model = parse_command_line_args()

    output_dir = 'output/'
    log_dir = output_dir + 'logs/'
    model_dir = output_dir + 'models/'
    make_dir(output_dir)
    make_dir(log_dir)
    make_dir(model_dir)
    ppo_path = model_dir + 'ppo_model_cartpole'
    env_id = 'CartPole-v1'

    # Step 2 - Train the model  
    if train_model == True:
        n_steps = 50000
        env_train = gym.make(id=env_id)
        env_train = DummyVecEnv([lambda: env_train])
        model = PPO('MlpPolicy', env_train, verbose=1, tensorboard_log=log_dir)
        model.learn(total_timesteps=n_steps)
        model.save(ppo_path)
        env_train.close()
    
    # Step 3 - Evaluate the model
    env_eval = gym.make(id=env_id)
    model = PPO.load(ppo_path)
    evaluate_policy(model, env_eval, n_eval_episodes=5)
    env_eval.close()

    n_episodes = 5
    env = gym.make(id=env_id, render_mode='human')
    for i in range(n_episodes):
        terminated = False
        state, _ = env.reset()
        score = 0

        while terminated == False:
            env.render()
            action, _ = model.predict(state)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
        print('Episode {0}: {1}'.format(i, score))
    env.close()

    return 1

if __name__ == '__main__':
    try:
        status = main()
    except Exception as ex:
        raise ex

