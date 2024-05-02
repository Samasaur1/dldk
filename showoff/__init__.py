import gymnasium as gym
from gymnasium.utils.play import play
from sys import argv, exit

def main():
    if len(argv) == 1:
        print("You must choose the strategy to use")
        print("Options: random,")
        exit(1)

    file_name = f"{argv[1]}.pt" if len(argv) == 2 else argv[2]

    env = gym.make("ALE/DonkeyKong-v5", render_mode='human')

    match argv[1]:
        case "random":
            import random

            if len(argv) > 2:
                print("Setting random seed to 0")
                random.seed(0)
                env.unwrapped.seed(0)
                env.action_space.seed(0)

            def model(state):
                return env.action_space.sample()
        case "q":
            model = QModel()
            model.load_state_dict(torch.load(file_name))
        case "qq":
            model = DoubleQ()
            model.load_state_dict(torch.load(file_name))
        case "ac":
            model = Actor()
            model.load_state_dict(torch.load(file_name))

    state, info = env.reset()
    reward = 0
    while True:
        action = model(state)
        ns, r, term, trunc, info = env.step(action)
        state += ns
        reward += r
        if term:
            print(f"Reward: {reward}")
            break

if __name__ == "__main__":
    main()
