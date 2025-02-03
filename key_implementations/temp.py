import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import os
import argparse
from typing import List
from maddpg import MLPNetwork


class Agent:
    def __init__(self, obs_dim, act_dim, global_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim)
        self.target_actor = MLPNetwork(obs_dim, act_dim)
        self.critic = MLPNetwork(global_dim, 1)
        self.target_critic = MLPNetwork(global_dim, 1)
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
        for agent_id in self.env.agents:
            self.agents[agent_id] = Agent(
                self.obs_dim, self.act_dim, global_obs_act_dim, actor_lr, critic_lr
            )
        self.buffer = Buffer(capacity, self.obs_dim, self.act_dim)
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
            self.buffer.add(o, a, r, next_o, d)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

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
