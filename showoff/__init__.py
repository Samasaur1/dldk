import gymnasium as gym
from gymnasium.utils.play import play
from sys import argv, exit

def main():
    if len(argv) == 1:
        print("You must choose the strategy to use")
        print("Options: random,")
        exit(1)

    file_name = f"{argv[1]}.pt" if len(argv) == 2 else argv[2]

    match argv[1]:
        case "random":
            import random

            env = gym.make("ALE/DonkeyKong-v5", render_mode='human')

            if len(argv) > 2:
                random.seed(0)
                env.seed(0)

            env.reset()
            ep_reward = 0
            while True:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated:
                    print(f"Reward: {ep_reward}")
                    break
        case "q":
            env = gym.make("ALE/DonkeyKong-v5", render_mode='human')

            model = QModel()
            model.load_state_dict(torch.load(file_name))

            state, info = env.reset()
            ep_reward = 0
            while True:
                action = model(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                ep_reward += reward
                if terminated:
                    print(f"Final reward: {ep_reward}")
                    break

if __name__ == "__main__":
    main()
