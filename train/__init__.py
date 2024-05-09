import gymnasium as gym
from gymnasium.utils.play import play
from sys import argv, exit

import torch
from torch import nn, optim
import numpy as np
import random
from tqdm import tqdm
from torch.distributions.categorical import Categorical

from models.q import QNetwork
from models.q import ReplayMemory

def main():
    if len(argv) == 1:
        print("You must choose the strategy to train")
        print("Options: q, qq, ac")
        exit(1)

    lr = 5e-4
    epochs = 1000
    batch_size = 128
    gamma = 0.99
    capacity = 10000
    tau = 0.005
    eps_start=0.9
    eps_end=0.5
    eps_rate=2000
    update_frequency = 100
    save_frequency = 5
    gae_lambda = 0.95

    if len(argv) == 2:
        file_name = f"{argv[1]}-lr-{lr}-epochs-{epochs}-batch_size-{batch_size}-gamma-{gamma}-capacity-{capacity}-tau-{tau}-eps_start-{eps_start}-eps_end-{eps_end}-eps_rate-{eps_rate}-update_frequency-{update_frequency}-commit-XXXXXX-machine-XXXX-model-N.pt"
    else:
        file_name = argv[2]

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Device: {device}")

    env = gym.make("ALE/DonkeyKong-v5", obs_type="grayscale")
    env = gym.wrappers.FrameStack(env, 4)

    match argv[1]:
        case "q":
            policy = QNetwork(env.action_space.n).to(device)
            opt = optim.Adam(policy.parameters(), lr=lr)

            memory = ReplayMemory(capacity)
            loss = nn.MSELoss()

            global_step = 0

            rng = np.random.default_rng()

            for i in tqdm(range(epochs)):
                state, info = env.reset()
                done = False

                while not done:
                    if rng.random() < eps_end + (eps_end - eps_start) * np.exp(-global_step / eps_rate):
                        action = env.action_space.sample()
                    else:
                        qs = policy(torch.Tensor(np.array(state)).to(device))
                        mx = qs.argmax(dim=1)
                        action = mx.cpu().numpy()
                        action = action[0] # I don't understand why this is necessary
                    global_step += 1
                    nstate, reward, term, trunc, _ = env.step(action)
                    memory.push(state, action, reward, nstate, term)
                    state = nstate

                    # Update the policy network
                    if len(memory) >= batch_size:
                        if global_step % update_frequency == 0:
                            batch = memory.sample(batch_size)
                            st_batch, act_batch, r_batch, nst_batch, t_batch = zip(*batch)
                            st_batch = torch.Tensor(np.array(st_batch)).to(device)
                            act_batch = torch.Tensor(act_batch).to(device).type(torch.int64).unsqueeze(dim=1)
                            r_batch = torch.Tensor(r_batch).to(device)
                            nst_batch = torch.Tensor(np.array(nst_batch)).to(device)
                            t_batch = torch.Tensor(t_batch).to(device).int()

                            target_max, _other = policy(nst_batch).max(dim=1)
                            target_max[t_batch] = 0
                            td_target = r_batch.flatten() + gamma * target_max.reshape((-1,128))
                            old_val = policy(st_batch).gather(1, act_batch).squeeze()
                            loss_val = loss(td_target, old_val)

                            opt.zero_grad()
                            loss_val.backward()
                            opt.step()

                            if global_step % (update_frequency * save_frequency) == 0:
                                torch.save(policy.state_dict(), file_name)

                    done = term or trunc

            torch.save(policy.state_dict(), file_name)
        case "qq":
            policy = QNetwork(env.action_space.n).to(device)
            opt = optim.Adam(policy.parameters(), lr=lr)
            policy = QNetwork(env.action_space.n).to(device)
            policy.load_state_dict(policy.state_dict())

            memory = ReplayMemory(capacity)
            loss = nn.MSELoss()

            global_step = 0

            rng = np.random.default_rng()

            for i in tqdm(range(epochs)):
                state, info = env.reset()
                done = False

                while not done:
                    if rng.random() < eps_end + (eps_end - eps_start) * np.exp(-global_step / eps_rate):
                        action = env.action_space.sample()
                    else:
                        qs = policy(torch.Tensor(np.array(state)).to(device))
                        mx = qs.argmax(dim=1)
                        action = mx.cpu().numpy()
                        action = action[0] # I don't understand why this is necessary
                    global_step += 1
                    nstate, reward, term, trunc, _ = env.step(action)
                    memory.push(state, action, reward, nstate, term)
                    state = nstate

                    # Update the policy network
                    if len(memory) >= batch_size:
                        if global_step % update_frequency == 0:
                            batch = memory.sample(batch_size)
                            st_batch, act_batch, r_batch, nst_batch, t_batch = zip(*batch)
                            st_batch = torch.Tensor(np.array(st_batch)).to(device)
                            act_batch = torch.Tensor(act_batch).to(device).type(torch.int64).unsqueeze(dim=1)
                            r_batch = torch.Tensor(r_batch).to(device)
                            nst_batch = torch.Tensor(np.array(nst_batch)).to(device)
                            t_batch = torch.Tensor(t_batch).to(device).int()

                            target_max, _other = policy(nst_batch).max(dim=1)
                            target_max[t_batch] = 0
                            td_target = r_batch.flatten() + gamma * target_max.reshape((-1,128))
                            old_val = policy(st_batch).gather(1, act_batch).squeeze()
                            loss_val = loss(td_target, old_val)

                            opt.zero_grad()
                            loss_val.backward()
                            opt.step()

                            p_state_dict = policy.state_dict()
                            t_state_dict = policy.state_dict()
                            for key in p_state_dict:
                                t_state_dict[key] = p_state_dict[key] * tau + t_state_dict[key] * (1 - tau)
                            policy.load_state_dict(t_state_dict)

                            if global_step % (update_frequency * save_frequency) == 0:
                                torch.save(policy.state_dict(), file_name)

                    done = term or trunc

            torch.save(policy.state_dict(), file_name)
        case "ac":
            def wrap(layer, std=np.sqrt(2), bias=0.0):
                torch.nn.init.orthogonal_(layer.weight, std)
                torch.nn.init.constant_(layer.bias, bias)
                return layer

            class Agent(nn.Module):
                def __init__(self, num_actions):
                    super().__init__()
                    self.conv = nn.Sequential(
                        wrap(nn.Conv2d(4, 32, 8, stride=4)),
                        nn.ReLU(),
                        wrap(nn.Conv2d(32, 64, 4, stride=2)),
                        nn.ReLU(),
                        wrap(nn.Conv2d(64, 64, 3, stride=1)),
                        nn.ReLU(),
                    )
                    self.fc = nn.Sequential(
                        wrap(nn.Linear(352, 512)),
                        nn.ReLU(),
                    )
                    self.actor = wrap(nn.Linear(512, num_actions), std=0.01)
                    self.critic = wrap(nn.Linear(512, 1), std=1)

                def get_value(self, x):
                    x = self.conv(x)
                    x = x.view(-1, 352)
                    x = self.fc(x)
                    return self.critic(x)

                def get_action_and_value(self, x, action=None):
                    x = self.conv(x)
                    x = x.view(-1, 352)
                    x = self.fc(x)
                    logits = self.actor(x)
                    probs = Categorical(logits=logits)
                    if action is None:
                        action = probs.sample()
                    return action, probs.log_prob(action), probs.entropy(), self.critic(x)

            agent = Agent(env.action_space.n).to(device)
            opt = optim.Adam(agent.parameters(), lr=lr, eps=1e-5) # 1e-5 hardcoded?

            # storage setup
            MEMORY_LENGTH = 128
            states = torch.zeros((MEMORY_LENGTH, 1) + env.observation_space.shape).to(device)
            actions = torch.zeros((MEMORY_LENGTH, 1) + env.action_space.shape).to(device)
            logprobs = torch.zeros((MEMORY_LENGTH, 1)).to(device)
            rewards = torch.zeros((MEMORY_LENGTH, 1)).to(device)
            dones = torch.zeros((MEMORY_LENGTH, 1)).to(device)
            values = torch.zeros((MEMORY_LENGTH, 1)).to(device)

            global_step = 0

            for i in tqdm(range(epochs)):
                state, info = env.reset()
                state = torch.Tensor(np.array(state)).to(device)
                done = False

                while not done:
                    for step in range(MEMORY_LENGTH):
                        global_step += 1
                        states[step] = state
                        dones[step] = done

                        action, logprob, _, value = agent.get_action_and_value(state)
                        # all the [0] s should not be necessary
                        values[step] = value.flatten()[0]
                        actions[step] = action[0]
                        logprobs[step] = logprob[0]

                        nstate, reward, term, trunc, _ = env.step(action[0].cpu().numpy())
                        done = term or trunc
                        rewards[step] = torch.tensor(reward).to(device).view(-1)
                        state = torch.Tensor(np.array(nstate)).to(device)

                        _highest = step
                        if done:
                            break
                    for i in range(_highest + 1, MEMORY_LENGTH):
                        states[i] = states[(2 * _highest) - i]
                        values[i] = values[(2 * _highest) - i]
                        dones[i] = dones[(2 * _highest) - i]
                        actions[i] = actions[(2 * _highest) - i]
                        logprobs[i] = logprobs[(2 * _highest) - i]
                        rewards[i] = rewards[(2 * _highest) - i]

                    next_value = agent.get_value(state).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(MEMORY_LENGTH)):
                        if t == MEMORY_LENGTH - 1:
                            nextnonterminal = 1.0 - done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam # TODO: breaks here
                    returns = advantages + values


                    # flatten the batch
                    st_batch = states.reshape((-1,) + env.observation_space.shape)
                    lp_batch = logprobs.reshape(-1)
                    act_batch = actions.reshape((-1,) + env.action_space.shape)
                    adv_batch = advantages.reshape(-1)
                    red_batch = returns.reshape(-1)
                    val_batch = values.reshape(-1)


                    # Optimizing the policy and value network
                    batch_indices = np.arange(MEMORY_LENGTH)
                    clipfracs = []
                    for _ in range(4):
                        np.random.shuffle(batch_indices)
                        for start in range(0, MEMORY_LENGTH, 32):
                            end = start + 32
                            batch_indices_batch = batch_indices[start:end]


                            _, newlogprob, entropy, newvalue = agent.get_action_and_value(st_batch[batch_indices_batch], act_batch.long()[batch_indices_batch])
                            logratio = newlogprob - lp_batch[batch_indices_batch]
                            ratio = logratio.exp()


                            with torch.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]


                            mb_advantages = adv_batch[batch_indices_batch]
                            if args.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)


                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()


                            # Value loss
                            newvalue = newvalue.view(-1)
                            v_loss = 0.5 * ((newvalue - red_batch[batch_indices_batch]) ** 2).mean()


                            entropy_loss = entropy.mean()
                            loss_val = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef


                            opt.zero_grad()
                            loss_val.backward()
                            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                            opt.step()


                        if args.target_kl is not None and approx_kl > args.target_kl:
                            break

                    # save_frequency doesn't make sense in this structure
                    torch.save(agent.state_dict(), file_name)
                
            torch.save(agent.state_dict(), file_name)
        case _:
            print("Unknown learning strategy!")
            print("Options: q, qq, ac")
            exit(1)
