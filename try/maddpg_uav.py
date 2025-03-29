import torch
import numpy as np
import torch.nn.functional as F
from cnn import  ActorNetwork as CNNActorNetwork
from agents import ActorNetwork, CriticNetwork, soft_update, GaussianNoise
from buffer import ReplayBuffer


class MADDPG:
    def __init__(self, num_agents, obs_shape, action_dim, hidden_dim=160, gamma=0.83, tau=0.01, device="cpu", use_cnn=True):
        self.num_agents = num_agents
        self.use_cnn = use_cnn
        self.device = device

        if use_cnn:
            # CNN :
            self.obs_dim = obs_shape
            self.grid_width, self.grid_height, self.input_channels = obs_shape
            self.actors = [CNNActorNetwork(input_channels=self.input_channels, action_dim=action_dim, hidden_dim=hidden_dim).to(device) for _ in range(num_agents)]
            self.target_actors = [CNNActorNetwork(input_channels=self.input_channels, action_dim=action_dim, hidden_dim=hidden_dim).to(device) for _ in range(num_agents)]

            self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.001) for actor in self.actors]

            sample_input = torch.zeros(1, self.input_channels, self.grid_width, self.grid_height).to(device)
            feature_dim = self.actors[0].cnn(sample_input).shape[1] * num_agents
            action_dim_critic = action_dim * num_agents

            self.critics = [CriticNetwork(feature_dim, action_dim_critic, hidden_dim).to(device) for _ in range(num_agents)]
            self.target_critics = [CriticNetwork(feature_dim, action_dim_critic, hidden_dim).to(device) for _ in range(num_agents)]
            self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=0.002) for critic in self.critics]
        else:
            # Vector observations :
            self.obs_dim = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
            self.total_obs_dim = num_agents * self.obs_dim
            self.total_action_dim = num_agents * action_dim
            self.actors = [ActorNetwork(self.obs_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
            self.target_actors = [ActorNetwork(self.obs_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
            self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.001) for actor in self.actors]
            self.critics = [CriticNetwork(self.total_obs_dim, self.total_action_dim, hidden_dim).to(device) for _ in range(num_agents)]
            self.target_critics = [CriticNetwork(self.total_obs_dim, self.total_action_dim, hidden_dim).to(device) for _ in range(num_agents)]
            self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=0.002) for critic in self.critics]

        
        self._init_target_networks()

        # Replay buffer
        self.buffer = ReplayBuffer(max_size=100000, num_agents=num_agents, obs_dim=self.obs_dim, action_dim=action_dim)

        # Noise for exploration
        self.noise = [GaussianNoise(action_dim) for _ in range(num_agents)]

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau

    def _init_target_networks(self):
        for target_actor, actor in zip(self.target_actors, self.actors):
            target_actor.load_state_dict(actor.state_dict())
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

    def select_action(self, obs, noise=True):
        """
        For CNN: obs is a list of grids with shape (num_agents, width, height, channels)
        For MLP: obs is a list of vectors with shape (num_agents, obs_dim)

        Returns: actions (num_agents, action_dim)
        """
        actions = []
        for i, actor in enumerate(self.actors):
            if self.use_cnn:
                # Grid observation processing
                # Convert numpy array to tensor and handle channel ordering
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).to(self.device)
                # CNN expects (batch, channels, height, width)
                if len(obs_tensor.shape) == 3:  # (height, width, channels)
                    obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, channels, height, width)
            else:
                # Vector observation processing
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).to(self.device)

            action = actor(obs_tensor).detach().cpu().numpy()
            if noise:
                action += self.noise[i].sample()
            actions.append(np.clip(action, -1, 1))
            
        return np.array(actions)

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return  # Not enough samples

        for i in range(self.num_agents):
            # Sample from buffer
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.buffer.sample(batch_size)

            # Convert to torch tensors
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
            act_tensor = torch.tensor(act_batch, dtype=torch.float32).to(self.device)
            next_obs_tensor = torch.tensor(next_obs_batch, dtype=torch.float32).to(self.device)
            rew_tensor = torch.tensor(rew_batch, dtype=torch.float32).to(self.device)
            done_tensor = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

            if self.use_cnn:
                # Process observations through CNN for each agent to get features
                all_obs_features = []
                all_next_obs_features = []

                for j in range(self.num_agents):
                    # Process current observations
                    # Reshape for CNN: (batch, agent, height, width, channels) -> (batch, channels, height, width)
                    agent_obs = obs_tensor[:, j].permute(0, 3, 1, 2)
                    agent_features = self.actors[j].cnn(agent_obs)
                    all_obs_features.append(agent_features)

                    # Process next observations
                    agent_next_obs = next_obs_tensor[:, j].permute(0, 3, 1, 2)
                    agent_next_features = self.target_actors[j].cnn(agent_next_obs)
                    all_next_obs_features.append(agent_next_features)

                # Concatenate features from all agents
                obs_flat = torch.cat(all_obs_features, dim=1)
                next_obs_flat = torch.cat(all_next_obs_features, dim=1)
            else:
                # For vector observations, just flatten
                obs_flat = obs_tensor.view(batch_size, -1)
                next_obs_flat = next_obs_tensor.view(batch_size, -1)

            act_flat = act_tensor.view(batch_size, -1)

            # Get next actions from target actors
            next_actions = []
            for j, target_actor in enumerate(self.target_actors):
                if self.use_cnn:
                    # Process grid observation
                    next_obs_j = next_obs_tensor[:, j].permute(0, 3, 1, 2)
                    next_agent_action = target_actor(next_obs_j)
                else:
                    # Process vector observation
                    next_agent_action = target_actor(next_obs_tensor[:, j, :])

                next_actions.append(next_agent_action)

            next_actions_tensor = torch.stack(next_actions, dim=1)
            next_actions_flat = next_actions_tensor.view(batch_size, -1)

            # Calculate target Q-value
            with torch.no_grad():
                target_q = self.target_critics[i](next_obs_flat, next_actions_flat)
                target_value = rew_tensor[:, i].unsqueeze(1) + self.gamma * target_q * (1 - done_tensor[:, i].unsqueeze(1))

            # Current Q-value
            current_q = self.critics[i](obs_flat, act_flat)

            # Critic loss
            critic_loss = F.mse_loss(current_q, target_value)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Create action inputs where only the current agent's action comes from the actor
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    if self.use_cnn:
                        # Process grid observation
                        obs_i = obs_tensor[:, i].permute(0, 3, 1, 2)
                        current_agent_action = self.actors[i](obs_i)
                    else:
                        # Process vector observation
                        current_agent_action = self.actors[i](obs_tensor[:, i, :])

                    current_actions.append(current_agent_action)
                else:
                    current_actions.append(act_tensor[:, j, :].detach())

            current_actions_tensor = torch.stack(current_actions, dim=1)
            current_actions_flat = current_actions_tensor.view(batch_size, -1)

            # Policy gradient
            actor_loss = -self.critics[i](obs_flat, current_actions_flat).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # Soft update target networks
            soft_update(self.target_critics[i], self.critics[i], self.tau)
            soft_update(self.target_actors[i], self.actors[i], self.tau)

        # Decay noise scale after each update
        self.decay_noise()

    def store(self, obs, actions, rewards, next_obs, dones):
        self.buffer.add(obs, actions, rewards, next_obs, dones)

    def decay_noise(self):
        for n in self.noise:
            n.decay()

    def reset_noise(self):
        for n in self.noise:
            n.reset()
