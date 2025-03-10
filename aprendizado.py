import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3", render_mode="rgb_array")

model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    train_freq=4,
    target_update_interval=1000,
)

print("Iniciando treinamento...")
model.learn(total_timesteps=int(200_000), progress_bar=True)

model.save("dqn_lunar")
print("Modelo salvo como dqn_lunar.zip")

del model

model = DQN.load("dqn_lunar", env=env)
print("Modelo carregado!")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Recompensa média: {mean_reward:.2f} ± {std_reward:.2f}")

obs, _ = env.reset()
rewards = []

for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    rewards.append(reward)
    env.render()

    if done:
        obs, _ = env.reset()

env.close()
