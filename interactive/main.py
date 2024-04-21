import gymnasium as gym
from gymnasium.utils.play import play

play(gym.make("ALE/DonkeyKong-v5", render_mode='rgb_array'))
