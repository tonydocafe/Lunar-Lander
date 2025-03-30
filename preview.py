import gymnasium as gym
from stable_baselines3 import DQN
import threading
import sys
import pygame  

env = gym.make("LunarLander-v3", render_mode="human")
model_path = "dqn_lunar.zip"
model = DQN.load(model_path, env=env)

def run_model():
    obs, _ = env.reset()
    
    try:
        for _ in range(1000):  
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, _, _ = env.step(action)
            env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt  

            if dones:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        env.close()
        pygame.quit()
        sys.exit(0)  

running = True

while running:
    run_model()

    enter_pressed = threading.Event()

    def wait_for_input():
        input("Press ENTER to exit...")
        enter_pressed.set()

    input_thread = threading.Thread(target=wait_for_input, daemon=True)
    input_thread.start()

    input_thread.join(timeout=3)  

    if enter_pressed.is_set():
        running = False  
        env.close()
        pygame.quit()
