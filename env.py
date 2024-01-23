import gym
from gym import spaces
import numpy as np
import torch
from DOI.doi_model import MLPModel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime


class CustomEnvironment(gym.Env):
    def __init__(self, model):
        super(CustomEnvironment, self).__init__()

        self.model = model

        # Define the action space (DOI values)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # Define the observation space (Efficiency values)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Initialize the state (initial Inf values)
        self.state = np.array([1.0, 1.0])

    def reset(self):
        # Reset the environment to the initial state
        self.state = np.array([1.0, 1.0])
        return self.state

    def step(self, action):
        # Execute the model with the provided DOI values
        doi_input = np.expand_dims(action, axis=0)
        input_tensor = torch.Tensor(doi_input)
        with torch.no_grad():
            eff_output = self.model(input_tensor)
        eff_output_unscaled = eff_output.numpy().flatten()

        # Update the state with the new efficiency values
        self.state = eff_output_unscaled

        # Calculate reward (higher reward for both efficiencies closer to 1)
        reward = -np.sum(np.abs(1 - eff_output_unscaled))

        # Check if the episode is done (for simplicity, you may define your own termination condition)
        done = False

        return self.state, reward, done, {}

def main():
    input_size = 2  # 输入特征数量
    hidden_size = 10  # 隐藏层神经元数量
    output_size = 2  # 输出变量数量（Eff_NH4 和 Eff_TP）

    # 加载最低 loss 的模型
    best_model = MLPModel(input_size, hidden_size, output_size)
    best_model.load_state_dict(torch.load('best_model.pth'))

    # Example of using the CustomEnvironment
    env = CustomEnvironment(best_model)
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the PPO model
    model.learn(total_timesteps=10000)

    # # Evaluate the trained model
    # mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    # print(f"Mean reward: {mean_reward}")

    # Example of using the trained PPO model to interact with the environment
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        print(f'reward{reward}')
        # env.render()

if __name__ == '__main__':
    main()
