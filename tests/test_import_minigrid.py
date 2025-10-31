import gymnasium as gym
import minigrid
from sar531.src.SAREnv import SAREnv


def test_import_minigrid():
    env = SAREnv()
    obs, info = env.reset(seed=100)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print("Successfully imported and ran MiniGrid environment.")
