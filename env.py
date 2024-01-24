import gym
from gym import spaces
import numpy as np
import torch
import matplotlib.pyplot as plt

from DOI.doi_model import MLPModel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime


class CustomEnvironment(gym.Env):
    def __init__(self, model):
        super(CustomEnvironment, self).__init__()

        self.model = model

        self.input_state = None

        # Define the action space (DOI values)
        self.action_space = spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32)

        # Define the observation space (Efficiency values)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Initialize the state (initial Inf values)
        self.state = np.random.uniform(low=0.0, high=4.0, size=(2,))

        # Init the current_step
        self.current_step = 0

    def set_input_state(self, input_state):
        # Set value of input_state
        self.input_state = input_state

    def reset(self):
        # Reset the environment to the initial state
        if self.input_state is not None:
            self.state = np.array(self.input_state)
        else:
            self.state = np.random.uniform(low=0.0, high=4.0, size=(2,))
        return self.state

    def step(self, action):
        doi_input = np.concatenate((self.state, action), axis=0)
        input_tensor = torch.Tensor(doi_input)
        with torch.no_grad():
            eff_output = self.model(input_tensor)
        eff_output_unscaled = eff_output.numpy().flatten()

        # Update current step
        self.current_step += 1

        # Update the state with the new efficiency values
        self.state = eff_output_unscaled

        # Calculate reward (higher reward for both efficiencies closer to 1)
        reward = -np.sum(np.abs(1 - eff_output_unscaled))

        # Check if the episode is done (for simplicity, you may define your own termination condition)
        if self.current_step > 10:
            done = True
        else:
            done = False
        # done = False

        return self.state, reward, done, {}

def main():
    input_size = 3  # 输入特征数量
    hidden_size1 = 1024  # 隐藏层神经元数量
    hidden_size2 = 1024
    output_size = 2  # 输出变量数量（Eff_NH4 和 Eff_TP）

    # 加载最低 loss 的模型
    best_model = MLPModel(input_size, hidden_size1, hidden_size2, output_size)
    best_model.load_state_dict(torch.load('best_model.pth'))

    # Example of using the CustomEnvironment
    env = CustomEnvironment(best_model)

    # 手动修改state值
    env.set_input_state([0.5, 0.5])

    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=1, n_steps=1024)

    # Train the PPO model
    model.learn(total_timesteps=20000)

    # # Evaluate the trained model
    # mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    # print(f"Mean reward: {mean_reward}")

    # Example of using the trained PPO model to interact with the environment
    init_state = [0.5, 0.5]
    obs = env.reset()

    # save reward value
    rewards = []

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        rewards.append(reward)
        print(f'reward{reward}')
        # env.render()

    # 可视化
    plt.plot(rewards, label='Reward', marker='o', linestyle='', markersize=1)
    plt.axhline(0, color='r', linestyle='--', label='Target Reward')  # 添加目标值为0的水平线
    plt.xlabel('Episode Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
