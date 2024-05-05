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
    gamma = 0.99
    capacity = 10000
    tau = 0.005
    eps_start=0.9
    eps_end=0.5
    eps_rate=2000
    update_frequency = 1000

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Device: {device}")

    match argv[1]:
        case "q":
            env = gym.make("ALE/DonkeyKong-v5", obs_type="grayscale")
            env = gym.wrappers.FrameStack(env, 4)

            class QNetwork(nn.Module):
                def __init__(self, input_shape, num_actions):
                    # print(f"num_actions: {num_actions}")
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(4, 32, 8, stride=4),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 4, stride=2),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, stride=1),
                        nn.ReLU(),
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(352, 512),
                        nn.ReLU(),
                        nn.Linear(512, num_actions),
                    )
                    # self.model = nn.Sequential(
                    #     nn.Conv2d(4, 32, 8, stride=4),
                    #     nn.ReLU(),
                    #     nn.Conv2d(32, 64, 4, stride=2),
                    #     nn.ReLU(),
                    #     nn.Conv2d(64, 64, 3, stride=1),
                    #     nn.ReLU(),
                    #     nn.Flatten(),
                    #     nn.Linear(352, 512),
                    #     nn.ReLU(),
                    #     nn.Linear(512, num_actions),
                    # )

                def forward(self, x):
                    # return self.model(x / 255.0)
                    x = self.conv(x)
                    x = x.view(-1, 352)
                    x = self.fc(x)
                    return x

            policy = QNetwork(env.observation_space.shape, env.action_space.n).to(device)
            opt = optim.Adam(policy.parameters(), lr=lr)
            target = QNetwork(env.observation_space.shape, env.action_space.n).to(device)
            target.load_state_dict(policy.state_dict())

            memory = ReplayMemory(capacity)
            loss = nn.MSELoss()

            global_step = 0

            rng = np.random.default_rng()

            for i in tqdm(range(epochs)):
                state, info = env.reset()
                done = False

                while not done:
                    # print(f"frame {global_step}, play {i}")
                    if rng.random() < eps_end + (eps_end - eps_start) * np.exp(-global_step / eps_rate):
                        action = env.action_space.sample()
                    else:
                        # # action = torch.argmax(policy(torch.Tensor(state).to(device)), dim=1).cpu().numpy()
                        # # # action = policy(torch.tensor(state).float()).argmax()
                        # # # action = policy(torch.tensor(state).to(device).float()).argmax()
                        # # # action = policy(torch.tensor(state, dtype=torch.float)).argmax()
                        # print(state)
                        # print(type(state))
                        # print(state.shape)
                        qs = policy(torch.Tensor(state).to(device))
                        # # qs = policy(torch.from_numpy(state).to(device))
                        # print(qs)
                        # print(type(qs))
                        # print(qs.shape)
                        mx = qs.argmax(dim=1)
                        # print(mx)
                        # print(type(mx))
                        # print(mx.shape)
                        action = mx.cpu().numpy()
                        # print(action)
                        # print(type(action))
                        # print(action.shape)
                        action = action[0] # I don't understand why this is necessary
                    global_step += 1
                    nstate, reward, term, trunc, _ = env.step(action)
                    memory.push(state, action, reward, nstate, term)
                    state = nstate
                    # count += 1

                    # Update the policy network
                    if len(memory) >= batch_size:
                        if global_step % update_frequency == 0:
                            batch = memory.sample(batch_size)
                            st_batch, act_batch, r_batch, nst_batch, t_batch = zip(*batch)
                            st_batch = torch.Tensor(st_batch).to(device)
                            act_batch = torch.Tensor(act_batch).to(device).type(torch.int64).unsqueeze(dim=1)
                            r_batch = torch.Tensor(r_batch).to(device)
                            nst_batch = torch.Tensor(nst_batch).to(device)
                            t_batch = torch.Tensor(t_batch).to(device).int()

                            # with torch.no_grad():
                            #     target_max, _ = target(nst_batch).max(dim=1)
                            #     td_target = r_batch.flatten() + gamma * target_max * (1 - t_batch.flatten())
                            # old_val = policy(st_batch).gather(1, act_batch).squeeze()
                            # loss_val = loss(td_target, old_val)
                            # print(nst_batch.shape)
                            target_max, _other = target(nst_batch).max(dim=1)
                            # print(target_max.shape, _other.shape)
                            target_max[t_batch] = 0
                            # print(r_batch.shape)
                            # print(target_max.shape)
                            td_target = r_batch.flatten() + gamma * target_max.reshape((-1,128))
                            # print(st_batch.shape)
                            # print(act_batch.shape)
                            old_val = policy(st_batch).gather(1, act_batch).squeeze()
                            loss_val = loss(td_target, old_val)

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
        case "qq":
            exit(1)
        case "ac":
            exit(1)



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
