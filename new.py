import numpy as np
import torch
from torch import nn
import random


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
        if self.is_actor:  # Actor network outputs actions in range [0, 1]
            output = torch.sigmoid(self.layer3(x))
        else:
            output = self.layer3(x)
        return output


class Agent:
    def __init__(self, obs_dim, act_dim, global_dim):
        self.actor = MLPNetwork(obs_dim, act_dim, is_actor=True)
        self.target_actor = MLPNetwork(obs_dim, act_dim, is_actor=True)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = MLPNetwork(global_dim, 1)
        self.target_critic = MLPNetwork(global_dim, 1)
        self.target_critic.load_state_dict(self.critic.state_dict())


class Buffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self._index = 0
        self.size = 0
        self.obs_buffer = np.zeros((capacity, obs_dim))
        self.action_buffer = np.zeros((capacity, act_dim))
        self.reward_buffer = np.zeros(capacity)
        self.next_obs_buffer = np.zeros((capacity, obs_dim))
        self.done_buffer = np.zeros(capacity, dtype=bool)

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buffer[self._index] = obs
        self.action_buffer[self._index] = action
        self.reward_buffer[self._index] = reward
        self.next_obs_buffer[self._index] = next_obs
        self.done_buffer[self._index] = done
        self._index = (self._index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, indices):
        obs = self.obs_buffer[indices]
        action = self.action_buffer[indices]
        reward = self.reward_buffer[indices]
        next_obs = self.next_obs_buffer[indices]
        done = self.done_buffer[indices]
        obs = torch.from_numpy(obs).float()
        action = torch.from_numpy(action).float()
        reward = (reward - reward.mean()) / (reward.std() + 1e-7)  # normalization
        reward = torch.from_numpy(reward).float()
        next_obs = torch.from_numpy(next_obs).float()
        done = torch.from_numpy(done).float()
        return obs, action, reward, next_obs, done


class MADDPG:
    def __init__(self):
        self.batch_size = 2
        self.env_agents = ["agent_0", "agent_1", "agent_2"]
        self.gamma = 0.99
        obs_dim = 4
        act_dim = 3
        self.num_agents = 3
        global_obs_act_dim = self.num_agents * (obs_dim + act_dim)
        self.agents = {}
        self.buffers = {}
        for agent_id in self.env_agents:
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim)
            self.buffers[agent_id] = Buffer(10, obs_dim, act_dim)

    def sample(self):
        indices = np.random.choice(
            self.buffers["agent_0"].size, size=self.batch_size, replace=False
        )
        print("indices : ", indices)
        obs_dict, act_dict, rew_dict, done_dict = {}, {}, {}, {}
        glob_obs, glob_act, glob_next_obs, glob_next_act = [], [], [], []
        # Sample experience from buffers of all agents
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs_dict[agent_id] = o
            glob_obs.append(o)
            act_dict[agent_id] = a
            glob_act.append(a)
            glob_next_obs.append(n_o)
            rew_dict[agent_id] = r
            done_dict[agent_id] = d
            # Calculate next actions using target network and next states
            next_actions = self.agents[agent_id].target_actor(n_o).detach()
            glob_next_act.append(next_actions)
        print("obs_dict : ", obs_dict)
        print("act_dict : ", act_dict)
        print("glob_obs : ", glob_obs)
        print("glob_act : ", glob_act)
        print("glob_next_obs : ", glob_next_obs)
        print("glob_next_act : ", glob_next_act)
        return (
            obs_dict,
            act_dict,
            rew_dict,
            done_dict,
            glob_obs,
            glob_act,
            glob_next_obs,
            glob_next_act,
        )

    # Heart of the algorithm
    def learn(self):
        for agent_id, agent in self.agents.items():
            print(agent_id)
            obs, act, reward, done, glob_obs, glob_act, glob_next_obs, glob_next_act = (
                self.sample()
            )

            # Update critic
            out = torch.cat(glob_obs + glob_act, 1)
            print("out : ", out)
            q = agent.critic(out).squeeze(1)
            print("q : ", q)
            next_out = torch.cat(glob_next_obs + glob_next_act, 1)
            print("next_out : ", next_out)
            q_target = agent.target_critic(next_out).squeeze(1)
            print("q_target : ", q_target)
            # Squeeze : [batch_size, 1] -> [batch_size]

            y = reward[agent_id] + self.gamma * q_target * (1 - done[agent_id])
            print("y : ", y)

            # Update actor
            act[agent_id] = agent.actor(obs[agent_id]).detach()
            print("act : ", act[agent_id])
            glob_act = list(act.values())
            print("glob_act : ", glob_act)
            new_out = torch.cat(glob_obs + glob_act, 1)
            print("new_out : ", new_out)
            q = agent.critic(new_out).squeeze(1)
            print("q : ", q)

    def foo(self, steps=10):
        for i in range(steps):

            if i == 5:
                for agent_id, buffer in self.buffers.items():
                    print(f"{agent_id} : ")
                    print("obs_buffer : ", buffer.obs_buffer)
                    print("action_buffer : ", buffer.action_buffer)
                    print("reward_buffer : ", buffer.reward_buffer)
                    print("next_obs_buffer : ", buffer.next_obs_buffer)
                    print("done_buffer : ", buffer.done_buffer)
                self.learn()

            for agent_id in self.env_agents:
                self.buffers[agent_id].add(
                    np.random.rand(4),
                    np.random.rand(3),
                    random.random(),
                    np.random.rand(4),
                    False,
                )


if __name__ == "__main__":

    maddpg = MADDPG()
    maddpg.foo()
