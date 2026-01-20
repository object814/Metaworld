import gymnasium as gym
import sys
import time
import imageio
import numpy as np

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld

# choose task names
PICK_ENV = "pick-place-v3"
DRAWER_ENV = "drawer-open-v3"

def start_gui(env_name, random_actions=True):
    env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="human", camera_name="front")
    # env = ProprioImageObsWrapper(env, image_height=128, image_width=128)
    obs, info = env.reset()
    done = False
    try:
        while True:
            if random_actions:
                action = env.action_space.sample()
            else:
                action = np.array([0.02, -0.02, 0.01, 0.1])
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"EE Pos: {obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}, EE velocity: {obs[3]:.3f}, {obs[4]:.3f}, {obs[5]:.3f}, Gripper Val: {obs[6]:.3f}")
            done = terminated or truncated
            env.render()
            if done:
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("Exiting GUI.")
    finally:
        env.close()

if __name__ == "__main__":
    start_gui(PICK_ENV, random_actions=False)