import gymnasium as gym
from gymnasium.utils.play import play
from sys import argv, exit

import torch
from torch import nn
import numpy as np

from models.q import QNetwork

import random

def main():
    if len(argv) == 1:
        print("You must choose the strategy to use")
        print("Options: random,")
        exit(1)

    file_name = f"{argv[1]}.pt" if len(argv) == 2 else argv[2]

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Device: {device}")

    env = gym.make("ALE/DonkeyKong-v5", obs_type="grayscale", render_mode='human')
    env = gym.wrappers.FrameStack(env, 4)

    if len(argv) > 3:
        seed = int(argv[3])
        print(f"Setting random seed to {seed}")
        random.seed(seed)
        env.unwrapped.seed(seed)
        env.action_space.seed(seed)

    match argv[1]:
        case "random":
            def model(state):
                return env.action_space.sample()
        case "q":
            _model = QNetwork(env.action_space.n).to(device)
            print("Created model")
            _model.load_state_dict(torch.load(file_name))
            print("Loaded state dict")
            def model(state):
                qs = _model(torch.Tensor(np.array(state)).to(device))
                mx = qs.argmax(dim=1)
                action = mx.cpu().numpy()
                return action[0]
            print("Model function defined")
        case "qq":
            _model = QNetwork(env.action_space.n).to(device)
            print("Created model")
            _model.load_state_dict(torch.load(file_name))
            print("Loaded state dict")
            def model(state):
                qs = _model(torch.Tensor(np.array(state)).to(device))
                mx = qs.argmax(dim=1)
                action = mx.cpu().numpy()
                return action[0]
        case "ac":
            model = Actor()
            model.load_state_dict(torch.load(file_name))

    print("Starting game")
    state, info = env.reset()
    lives = 4
    reward = 0
    while True:
        action = model(state)
        ns, r, term, trunc, info = env.step(action)
        state = ns
        reward += r
        if r != 0:
            print(f"Step reward: {r}")
        if term:
            print(f"Total reward: {reward}")
            break
        if info['lives'] != lives:
            print("Lost life")
            print("Force jumping to start next life")
            ns, r, term, trunc, info = env.step(1)
            lives = info['lives']
            state = ns

if __name__ == "__main__":
    main()
