import gymnasium as gym
import minigrid


def test_import_minigrid():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    obs, info = env.reset(seed=100)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print("Successfully imported and ran MiniGrid environment.")
