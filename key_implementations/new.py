import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from pettingzoo.mpe import simple_spread_v3
from collections import deque
import random

# Set up the environment
env = simple_spread_v3.env()

# Hyperparameters
n_agents = 3  # Number of agents in the environment
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n  # Assuming discrete action space
hidden_dim = 128
gamma = 0.95  # Discount factor
tau = 0.01  # Soft target update
batch_size = 1024
replay_buffer_size = 1000000
lr_actor = 1e-4
lr_critic = 1e-3
epsilon = 0.1  # Exploration noise
train_interval = 10

# Experience Replay
class ReplayBuffer:
    def __init__(self, max_size=replay_buffer_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=-1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# MADDPG Agent
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, gamma, lr_actor, lr_critic, tau):
        self.actor = Actor(state_dim, action_dim, hidden_dim).cuda()
        self.target_actor = Actor(state_dim, action_dim, hidden_dim).cuda()
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).cuda()
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).cuda()
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
    
    def update_targets(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def act(self, state, noise=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
        action_probs = self.actor(state).detach().cpu().numpy().flatten()
        action = np.argmax(action_probs) + np.random.normal(0, noise)
        action = np.clip(action, 0, action_probs.shape[0] - 1)
        return int(action)

    def train(self, batch, all_agents, max_action):
        states, actions, rewards, next_states, next_actions, dones = batch
        
        states = torch.tensor(states, dtype=torch.float32).cuda()
        actions = torch.tensor(actions, dtype=torch.long).cuda()
        rewards = torch.tensor(rewards, dtype=torch.float32).cuda()
        next_states = torch.tensor(next_states, dtype=torch.float32).cuda()
        next_actions = torch.tensor(next_actions, dtype=torch.long).cuda()
        dones = torch.tensor(dones, dtype=torch.float32).cuda()

        # Update critic
        with torch.no_grad():
            next_q_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.squeeze(-1)
        
        current_q_values = self.critic(states, actions).squeeze(-1)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_targets()

# Initialize agents and replay buffer
agents = [MADDPGAgent(state_dim, action_dim, hidden_dim, gamma, lr_actor, lr_critic, tau) for _ in range(n_agents)]
replay_buffer = ReplayBuffer()

# Training loop
episodes = 10000
for episode in range(episodes):
    env.reset()
    episode_rewards = np.zeros(n_agents)
    done = False
    while not done:
        states = [env.observation(agent) for agent in env.agent_iter()]
        actions = []
        for agent in range(n_agents):
            action = agents[agent].act(states[agent], noise=epsilon)
            actions.append(action)
        actions = np.array(actions)
        
        # Apply actions and get next state and reward
        next_states, rewards, dones, _ = env.step(actions)
        episode_rewards += rewards
        
        # Store transition in replay buffer
        for agent in range(n_agents):
            next_actions = [agents[other].act(next_states[other], noise=epsilon) for other in range(n_agents)]
            transition = (states[agent], actions[agent], rewards[agent], next_states[agent], next_actions, dones[agent])
            replay_buffer.push(transition)
        
        # Train agents every few steps
        if replay_buffer.size() > batch_size and episode % train_interval == 0:
            batch = replay_buffer.sample(batch_size)
            for agent in agents:
                agent.train(batch, agents, action_dim)
    
    print(f"Episode {episode}, Rewards: {episode_rewards}")
