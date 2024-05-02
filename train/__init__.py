import gymnasium as gym
from gymnasium.utils.play import play
from sys import argv, exit

import torch
from torch import nn, optim
import numpy as np
import random
from tqdm import tqdm

from models.q import Q
from models.q import ReplayMemory
from models.q import ActorCnn

def main():
    if len(argv) == 1:
        print("You must choose the strategy to train")
        print("Options: ,")
        exit(1)

    file_name = f"{argv[1]}.pt" if len(argv) == 2 else argv[2]

    lr = 5e-4
    epochs = 1000
    batch_size = 128
    capacity = 10000
    tau = 0.005
    eps_start=0.9
    eps_end=0.5
    eps_rate=2000

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Device: {device}")

    match argv[1]:
        case "q":
            env = gym.make("ALE/DonkeyKong-v5")

            policy = Q(env.observation_space.shape, env.action_space.n)#.to(device)
            target = Q(env.observation_space.shape, env.action_space.n)#.to(device)

            target.load_state_dict(policy.state_dict())

            memory = ReplayMemory(capacity)
            opt = optim.Adam(policy.parameters(), lr=lr)
            loss = nn.MSELoss()

            global_step = 0

            rng = np.random.default_rng()

            for i in tqdm(range(epochs)):
                state, info = env.reset()
                done = False

                while not done:
                    if rng.random() < eps_end + (eps_end - eps_start) * np.exp(-global_step / eps_rate):
                        action = rng.integers(0, env.action_space.n)
                    else:
                        action = policy(torch.tensor(state).float()).argmax()
                        # action = policy(torch.tensor(state).to(device).float()).argmax()
                        # action = policy(torch.tensor(state, dtype=torch.float)).argmax()
                    global_step += 1
                    nstate, reward, term, trunc, _ = env.step(action)
                    memory.push(state, action, reward, nstate, term)
                    state = nstate
                    # count += 1

                    # Update the policy network
                    if len(memory) >= batch_size:
                        batch = memory.sample(batch_size)
                        st_batch, act_batch, r_batch, nst_batch, t_batch = zip(*batch)
                        st_batch = torch.tensor(np.array(st_batch)).float()
                        act_batch = torch.tensor(np.array(act_batch)).unsqueeze(dim=1)
                        r_batch = torch.tensor(np.array(r_batch)).float()
                        nst_batch = torch.tensor(np.array(nst_batch)).float()
                        t_batch = torch.tensor(np.array(t_batch))

                        # pred_vals is the predicted Q value of the sampled
                        # state-action pairs from the dataset
                        pred_vals = policy(st_batch).gather(1, act_batch).squeeze()
                        # pred_vals = policy(st_batch.to(device)).gather(1, act_batch).squeeze()

                        # pred_next_vals is the predicted value of the sampled next
                        # states. This is where we use the trick of setting the value
                        # of terminal states to zero.
                        pred_next_vals = target(nst_batch).max(dim=1).values
                        # pred_next_vals = target(nst_batch.to(device)).max(dim=1).values
                        pred_next_vals[t_batch] = 0

                        # expected_q is the right side of our loss from above.
                        expected_q = r_batch + gamma * pred_next_vals

                        # This part is just like what we've seen before.
                        loss_val = loss(pred_vals, expected_q)
                        opt.zero_grad()
                        loss_val.backward()
                        opt.step()

                    p_state_dict = policy.state_dict()
                    t_state_dict = target.state_dict()
                    for key in p_state_dict:
                        t_state_dict[key] = p_state_dict[key] * tau + t_state_dict[key] * (1 - tau)
                    target.load_state_dict(t_state_dict)

                    done = term or trunc

            torch.save(policy.state_dict(), file_name)



        # case "q":
        #     env = gym.make("ALE/DonkeyKong-v5")
        #
        #     policy = ActorCNN(env.observation_space.shape, env.action_space.n)
        #
        #     memory = ReplayMemory(capacity)
        #     opt = optim.Adam(policy.parameters(), lr=lr)
        #     loss = nn.MSELoss()
        #
        #     global_step = 0
        #
        #     for i in tqdm(range(epochs)):
        #         state, info = env.reset()
        #         done = False
        #
        #         while not done:
        #             if rng.random() < eps_end + (eps_end - eps_start) * np.exp(-global_step / eps_rate):
        #                 action = rng.integers(0, env.action_space.n)
        #             else:
        #                 action = policy(state)
        #                 # action = policy(torch.tensor(state, dtype=torch.float)).argmax()
        #             global_step += 1
        #
        #             next_state, reward, terminated, truncated, info = env.step(action)
        #             memory.push(state, action, reward, next_state, terminated)
        #
        #             # update agent
        #             if len(memory) >= batch_size:
        #                 batch = memory.sample(batch_size)
        #                 st_batch, act_batch, r_batch, nst_batch, t_batch = zip(*batch)
        #                 st_batch = torch.tensor(np.array(st_batch)).float()
        #                 act_batch = torch.tensor(np.array(act_batch)).unsqueeze(dim=1)
        #                 r_batch = torch.tensor(np.array(r_batch)).float()
        #                 nst_batch = torch.tensor(np.array(nst_batch)).float()
        #                 t_batch = torch.tensor(np.array(t_batch))
        #
        #             state = next_state
        #             done = terminated or truncated
