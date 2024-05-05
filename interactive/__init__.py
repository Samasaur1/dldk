import gymnasium as gym
from gymnasium.utils.play import play

def main():
    global total_reward
    total_reward = 0
    def callback(prev_obs, obs, action, rew, terminated, truncated, info):
        global total_reward
        total_reward += rew
        if rew != 0:
            print(f"Step reward: {rew}")
        if terminated:
            print(f"Total reward: {total_reward}")
            total_reward = 0
    play(gym.make("ALE/DonkeyKong-v5", render_mode='rgb_array'), callback=callback)

if __name__ == "__main__":
    main()
