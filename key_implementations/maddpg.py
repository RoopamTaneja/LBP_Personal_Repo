'''
agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded 
based on how far the closest agent is to each landmark 
(negative sum of the minimum distances). 
Locally, the agents are penalized if they collide with 
other agents (-1 for each collision).
'''

# https://pettingzoo.farama.org/environments/mpe/simple_spread/

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3

import os
import pickle
from typing import List


from sample import Agent, Buffer


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, episode_length, capacity, batch_size, actor_lr, critic_lr):
        self.env = simple_spread_v3.parallel_env(max_cycles=episode_length)
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
            self.buffers[agent_id] = Buffer(capacity, self.obs_dim, self.act_dim, "cpu")

        self.batch_size = batch_size

    def add(self, obs, action, reward, next_obs, termination, truncation):
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a= np.eye(self.act_dim)[a]

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

    def learn(self, batch_size, gamma):
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(
                list(next_obs.values()), list(next_act.values())
            )
            target_value = reward[agent_id] + gamma * next_target_critic_value * (
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

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def train(self, episodes):
        step = 0  # global step counter
        self.env.reset()
        # reward of each episode of each agent
        episode_rewards = {agent_id: np.zeros(episodes) for agent_id in self.env.agents}
        for episode in range(episodes):
            obs, _ = self.env.reset()
            agent_reward = {
                agent_id: 0 for agent_id in self.env.agents
            }  # agent reward of the current episode
            while self.env.agents:  # interact with the env for an episode
                step += 1
                if step < STEPS:
                    action = {
                        agent_id: self.env.action_space(agent_id).sample()
                        for agent_id in self.env.agents
                    }
                else:
                    action = self.select_action(obs)

                next_obs, reward, termination, truncation, _ = self.env.step(action)
                self.add(obs, action, reward, next_obs, termination, truncation)
                obs = next_obs

                for agent_id, r in reward.items():  # update reward
                    agent_reward[agent_id] += r

                if (
                    step >= STEPS and step % LEARN_INTERVAL == 0
                ):  # learn every few steps
                    self.learn(BATCH_SIZE, GAMMA)
                    self.update_target(TAU)


            # episode finishes
            for agent_id, r in agent_reward.items():  # record reward
                episode_rewards[agent_id][episode] = r

            if (episode + 1) % 100 == 0:  # print info every 100 episodes
                message = f"episode {episode + 1}, "
                sum_reward = 0
                for agent_id, r in agent_reward.items():  # record reward
                    message += f"{agent_id}: {r:>4f}; "
                    sum_reward += r
                message += f"sum reward: {sum_reward}"
                print(message)

        return episode_rewards

    def test(self, episodes):
        # reward of each episode of each agent
        self.env.reset()
        episode_rewards = {agent_id: np.zeros(episodes) for agent_id in self.env.agents}
        for episode in range(episodes):
            obs, _ = self.env.reset()
            agent_reward = {
                agent_id: 0 for agent_id in self.env.agents
            }  # agent reward of the current episode
            while self.env.agents:  # interact with the env for an episode
                actions = self.select_action(obs)
                next_obs, rewards, _, _, _ = self.env.step(actions)
                obs = next_obs

                for agent_id, reward in rewards.items():  # update reward
                    agent_reward[agent_id] += reward

            message = f"episode {episode + 1}, "
            # episode finishes, record reward
            for agent_id, reward in agent_reward.items():
                episode_rewards[agent_id][episode] = reward
                message += f"{agent_id}: {reward:>4f}; "
            print(message)
        return episode_rewards

    def save(self, reward, result_dir):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {
                name: agent.actor.state_dict() for name, agent in self.agents.items()
            },  # actor parameter
            os.path.join(result_dir, "model.pt"),
        )
        with open(
            os.path.join(result_dir, "rewards.pkl"), "wb"
        ) as f:  # save training data
            pickle.dump({"rewards": reward}, f)

    @classmethod
    def load(cls, dim_info, file):
        """init self using the model saved in `file`"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance


if __name__ == "__main__":

    ENV_NAME = "simple_spread_v3"
    # NUM_EPISODES = 3000
    NUM_TRAIN_EPISODES = 500
    NUM_TEST_EPISODES = 10

    EPISODE_LENGTH = 25
    LEARN_INTERVAL = 10
    STEPS = 5000
    TAU = 0.02
    GAMMA = 0.95
    BUFFER_CAPACITY = int(1e6)
    BATCH_SIZE = 1024
    ACTOR_LR = 0.01
    CRITIC_LR = 0.01

    # create folder to save result
    env_dir = os.path.join("./results", ENV_NAME)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f"{total_files + 1}")
    os.makedirs(result_dir)

    maddpg = MADDPG(
        EPISODE_LENGTH, BUFFER_CAPACITY, BATCH_SIZE, ACTOR_LR, CRITIC_LR
    )

    episode_rewards = maddpg.train(NUM_TRAIN_EPISODES)  # train
    maddpg.save(episode_rewards, result_dir)  # save model

    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[: i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1 : i + 1])
        return running_reward

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, NUM_TRAIN_EPISODES + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Result")
    plt.savefig(os.path.join(result_dir, "Training result"))

    # model_dir = os.path.join("./results", ENV_NAME, "1")
    # assert os.path.exists(model_dir)
    # maddpg = MADDPG.load(dim_info, os.path.join(model_dir, "model.pt"))

    episode_rewards = maddpg.test(NUM_TEST_EPISODES)  # test

    # testing finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, NUM_TEST_EPISODES + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent_id)
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Testing Result")
    # plt.savefig(os.path.join(model_dir, title))
