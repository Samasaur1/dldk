import gymnasium as gym
from gymnasium.utils.play import play

def main():
    play(gym.make("ALE/DonkeyKong-v5", render_mode='rgb_array'))

if __name__ == "__main__":
    main()
