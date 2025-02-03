# Pytorch implementation of MADDPG algorithm
# for simple_spread_v3 cooperative environment of PettingZoo MPE (prev OpenAI MPE)
# https://pettingzoo.farama.org/environments/mpe/simple_spread/
# Implemented for a parallel environment with continuous actions

# Agents must learn to cover all the landmarks while avoiding collisions.
# More specifically, all agents are globally rewarded
# based on how far the closest agent is to each landmark
# (negative sum of the minimum distances).
# Locally, the agents are penalized if they collide with
# other agents (-1 for each collision).

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import os
import argparse
from typing import List


class OUActionNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.mu = np.ones(action_dim) * mu
        self.theta = theta
        self.sigma = np.ones(action_dim) * sigma
        self.dt = dt
        self.reset()

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state

    def reset(self):
        self.state = np.copy(self.mu)


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, is_actor=False):
        super(MLPNetwork, self).__init__()
        self.is_actor = is_actor
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, input):
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float)
        x = nn.functional.relu(self.layer1(input))
        x = nn.functional.relu(self.layer2(x))
        if self.is_actor:
            output = torch.sigmoid(self.layer3(x))
        else:
            output = self.layer3(x)
        return output


class Agent:
    def __init__(self, obs_dim, act_dim, global_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim, is_actor=True)
        self.target_actor = MLPNetwork(obs_dim, act_dim, is_actor=True)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = MLPNetwork(global_dim, 1)
        self.target_critic = MLPNetwork(global_dim, 1)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class Buffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity

        self.obs_buffer = np.zeros((capacity, obs_dim))
        self.action_buffer = np.zeros((capacity, act_dim))
        self.reward_buffer = np.zeros(capacity)
        self.next_obs_buffer = np.zeros((capacity, obs_dim))
        self.done_buffer = np.zeros(capacity, dtype=bool)

        self._index = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, done):
        """add an experience to the memory"""
        self.obs_buffer[self._index] = obs
        self.action_buffer[self._index] = action
        self.reward_buffer[self._index] = reward
        self.next_obs_buffer[self._index] = next_obs
        self.done_buffer[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs_buffer[indices]
        action = self.action_buffer[indices]
        reward = self.reward_buffer[indices]
        next_obs = self.next_obs_buffer[indices]
        done = self.done_buffer[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.from_numpy(obs).float()  # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(
            action
        ).float()  # torch.Size([batch_size, action_dim])
        reward = torch.from_numpy(
            reward
        ).float()  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.from_numpy(next_obs).float()  # Size([batch_size, state_dim])
        done = torch.from_numpy(done).float()  # just a tensor with length: batch_size

        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size


class MADDPG:
    def __init__(
        self, episode_length, capacity, batch_size, gamma, tau, actor_lr, critic_lr
    ):
        self.env = simple_spread_v3.parallel_env(
            max_cycles=episode_length, continuous_actions=True
        )
        self.env.reset()
        first_agent = self.env.agents[0]
        self.obs_dim = self.env.observation_space(first_agent).shape[0]
        self.act_dim = self.env.action_space(first_agent).n
        global_obs_act_dim = self.env.num_agents * (self.obs_dim + self.act_dim)
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        for agent_id in self.env.agents:
            self.agents[agent_id] = Agent(
                self.obs_dim, self.act_dim, global_obs_act_dim, actor_lr, critic_lr
            )
            self.buffers[agent_id] = Buffer(capacity, self.obs_dim, self.act_dim)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def add(self, obs, action, reward, next_obs, termination, truncation):
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.act_dim)[a]

            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = termination[agent_id] or truncation[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers["agent_0"])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)

        return obs, act, reward, next_obs, done, next_act

    def select_action(self, obs):
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float()
            a = self.agents[agent].action(o)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax().item()
        return actions

    def learn(self):
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(self.batch_size)
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(
                list(next_obs.values()), list(next_act.values())
            )
            target_value = reward[agent_id] + self.gamma * next_target_critic_value * (
                1 - done[agent_id]
            )

            critic_loss = F.mse_loss(
                critic_value, target_value.detach(), reduction="mean"
            )
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], model_out=True)
            act[agent_id] = action
            actor_loss = -agent.critic_value(
                list(obs.values()), list(act.values())
            ).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)

    def update_targets(self):
        def soft_update(from_network, to_network):
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(self.tau * from_p.data + (1.0 - self.tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def train(self, episodes, initial_steps, learn_interval):
        step = 0
        self.env.reset()
        episode_rewards = {agent_id: np.zeros(episodes) for agent_id in self.env.agents}

        for epi in range(episodes):
            obs, _ = self.env.reset()
            curr_epi_rewards = {agent_id: 0 for agent_id in self.env.agents}

            while self.env.agents:
                step += 1
                if step < initial_steps:  # randomly explore initially
                    action = {
                        agent_id: self.env.action_space(agent_id).sample()
                        for agent_id in self.env.agents
                    }
                else:  # select action using policy
                    action = self.select_action(obs)

                next_obs, reward, term, trunc, _ = self.env.step(action)
                for agent_id in obs.keys():
                    self.buffers[agent_id].add(
                        obs[agent_id],
                        action[agent_id],
                        reward[agent_id],
                        next_obs[agent_id],
                        term[agent_id] or trunc[agent_id],
                    )
                obs = next_obs

                for agent_id, r in reward.items():
                    curr_epi_rewards[agent_id] += r

                if step >= initial_steps and step % learn_interval == 0:
                    self.learn()
                    self.update_targets()

            for agent_id, r in curr_epi_rewards.items():
                episode_rewards[agent_id][epi] = r

            if (epi + 1) % 100 == 0:
                message = f"Episode {epi + 1}, "
                sum_reward = 0
                for agent_id, reward in curr_epi_rewards.items():
                    message += f"{agent_id}: {reward:>4f}; "
                    sum_reward += reward
                message += f"Sum: {sum_reward}"
                print(message)

        return episode_rewards

    def test(self, episodes):
        self.env.reset()
        episode_rewards = {agent_id: np.zeros(episodes) for agent_id in self.env.agents}

        for epi in range(episodes):
            obs, _ = self.env.reset()
            curr_epi_rewards = {agent_id: 0 for agent_id in self.env.agents}

            while self.env.agents:
                action = self.select_action(obs)
                next_obs, reward, _, _, _ = self.env.step(action)
                obs = next_obs

                for agent_id, r in reward.items():
                    curr_epi_rewards[agent_id] += r

            for agent_id, r in curr_epi_rewards.items():
                episode_rewards[agent_id][epi] = r

            if (epi + 1) % 5 == 0:
                message = f"Episode {epi + 1}, "
                sum_reward = 0
                for agent_id, reward in curr_epi_rewards.items():
                    message += f"{agent_id}: {reward:>4f}; "
                    sum_reward += reward
                message += f"Sum: {sum_reward}"
                print(message)

        return episode_rewards


# saving actor parameters for each agent
def save_model(maddpg, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    torch.save(
        {name: agent.actor.state_dict() for name, agent in maddpg.agents.items()},
        os.path.join(result_dir, "model.pt"),
    )


def get_running_reward(arr: np.ndarray, window=100):
    running_reward = np.zeros_like(arr)
    for i in range(window - 1):
        running_reward[i] = np.mean(arr[: i + 1])
    for i in range(window - 1, len(arr)):
        running_reward[i] = np.mean(arr[i - window + 1 : i + 1])
    return running_reward


def plot_graphs(num_episodes, episode_rewards, result_dir, test=False):
    _, ax = plt.subplots()
    x = range(1, num_episodes + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        if test == False:
            ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    if test:
        title = "Testing"
    else:
        title = "Training"
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))


def load_model(maddpg, episode_length, file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    data = torch.load(file, weights_only=True)
    maddpg.env = simple_spread_v3.parallel_env(
        max_cycles=episode_length, continuous_actions=True
    )
    maddpg.env.reset()
    for agent_id, agent in maddpg.agents.items():
        agent.actor.load_state_dict(data[agent_id])


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train", action="store_true")
    # parser.add_argument("--test", action="store_true")
    # parser.add_argument("--local_ratio", type=float, default=0.5)
    # args = parser.parse_args()
    # env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=True, local_ratio=args.local_ratio)

    LEARN_INTERVAL = 100
    INITIAL_STEPS = 5000
    TAU = 0.02
    GAMMA = 0.95
    BUFFER_CAPACITY = 1000000
    BATCH_SIZE = 1024
    ACTOR_LR = 0.01
    CRITIC_LR = 0.01
    NUM_TRAIN_EPISODES = 3000
    NUM_TEST_EPISODES = 50
    TRAIN_EPISODE_LENGTH = 25
    TEST_EPISODE_LENGTH = 50

    maddpg = MADDPG(
        TRAIN_EPISODE_LENGTH,
        BUFFER_CAPACITY,
        BATCH_SIZE,
        GAMMA,
        TAU,
        ACTOR_LR,
        CRITIC_LR,
    )
    result_dir = os.path.join("./results")

    # if args.train:
    print("Training")
    episode_rewards = maddpg.train(NUM_TRAIN_EPISODES, INITIAL_STEPS, LEARN_INTERVAL)
    # save_model(maddpg, result_dir)
    # plot_graphs(NUM_TRAIN_EPISODES, episode_rewards, result_dir)

    # if args.test:
    print("Testing")
    # load_model(maddpg, TEST_EPISODE_LENGTH, os.path.join(result_dir, "model.pt"))
    episode_rewards = maddpg.test(NUM_TEST_EPISODES)
    # plot_graphs(NUM_TEST_EPISODES, episode_rewards, result_dir, test=True)
