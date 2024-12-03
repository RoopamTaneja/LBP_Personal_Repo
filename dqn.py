import gym
import numpy as np
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt


# Neural network architecture class
class QNetwork(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.q_values = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.q_values(x)


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# DQN Agent
class DQNAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        learning_rate=1e-3,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n

        # Two instances of neural network - Q network and target network
        self.q_network = QNetwork(self.input_size, self.output_size)
        self.target_network = QNetwork(self.input_size, self.output_size)

        # target network is identical to Q network initially
        self.target_network.set_weights(self.q_network.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.replay_buffer = ReplayBuffer(10000)

    def act(self, state):
        if random.random() < self.epsilon:  # random action
            return self.env.action_space.sample()

        # else pass to neural network to get action values and
        # choose optimal action
        state = np.expand_dims(
            state, axis=0
        )  # Make 1D to 2D array (to pass as argument)
        q_values = self.q_network(state)
        return np.argmax(q_values)

    def learn(self, batch_size):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        # Unpack batch into 5 separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert them into numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # The GradientTape is used to track the operations in the forward pass so that gradients can later be computed.
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values = tf.gather(q_values, actions, axis=1, batch_dims=1)

            next_q_values = tf.reduce_max(self.target_network(next_states), axis=1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

            # Mean Squared Error
            loss = tf.reduce_mean(tf.square(q_values - target_q_values))

        # Compute gradients and update weights of Q network
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Decaying epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Copy Q network to target network periodically
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())


# Train the agent
def train_dqn(agent, episodes=1000, batch_size=64):
    episode_rewards = []
    for episode in range(episodes):
        state = agent.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = agent.env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.learn(batch_size)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if episode % 10 == 0:
            agent.update_target_network()

        if episode % 100 == 0:
            print(
                f"Episode {episode}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}"
            )

    return episode_rewards


# Main execution
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)

    episode_rewards = train_dqn(agent)

    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Progress")
    plt.show()
