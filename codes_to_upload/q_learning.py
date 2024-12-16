import numpy as np
import gymnasium as gym
import random


class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lower_bounds = self.env.observation_space.low
        self.upper_bounds = self.env.observation_space.high
        num_bins = 10
        self.state_bins = []
        for i in range(len(self.upper_bounds)):
            self.state_bins.append(
                np.linspace(self.lower_bounds[i], self.upper_bounds[i], num_bins)
            )

        # Initialize the Q-table with zeros
        self.Q = np.zeros(
            [len(bins) + 1 for bins in self.state_bins] + [self.env.action_space.n]
        )

    def discretize_state(self, state):
        state = np.array(state)
        state = np.clip(state, self.lower_bounds, self.upper_bounds)
        state_indices = []
        for i, bins in enumerate(self.state_bins):
            index = np.digitize(state[i], bins) - 1
            index = np.clip(index, 0, len(bins) - 1)
            state_indices.append(index)
        return tuple(state_indices)

    def choose_action(self, state):  # Behaviour policy : Epsilon-greedy
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def choose_action_greedy(self, state):  # Target policy : Greedy
        return np.argmax(self.Q[state])

    # Q-learning algorithm
    def train_agent(self, num_train_episodes, max_steps):
        episode_rewards = []
        for episode in range(num_train_episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            total_reward = 0

            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)

                # Update Q-table using the Q-learning update rule
                self.Q[state][action] = self.Q[state][action] + self.alpha * (
                    reward
                    + self.gamma * np.max(self.Q[next_state])
                    - self.Q[state][action]
                )

                state = next_state  # Move to the next state
                total_reward += reward

                if done or truncated:  # If the episode ends
                    break

            episode_rewards.append(total_reward)

            # Print the progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(
                    f"Episode {episode + 1}/{num_train_episodes}, Total Reward: {total_reward}",
                    f"Average Reward (last 100): {avg_reward:.2f}, ",
                )

    def test_agent(self, num_test_episodes, max_steps):
        episode_rewards = []
        for episode in range(num_test_episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            total_reward = 0

            for _ in range(max_steps):
                action = self.choose_action_greedy(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)

                state = next_state
                total_reward += reward

                if done or truncated:
                    break

            episode_rewards.append(total_reward)
            print(f"Test Episode {episode + 1} Reward: {total_reward}")

        print(f"Average Test Reward: {np.mean(episode_rewards):.2f}")


if __name__ == "__main__":

    # Using CartPole for environment
    env = gym.make("CartPole-v1")

    # Q-learning parameters
    num_train_episodes = 1000
    max_steps = 200
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    num_test_episodes = 10

    agent = QLearningAgent(env, alpha, gamma, epsilon)
    agent.train_agent(num_train_episodes, max_steps)
    agent.test_agent(num_test_episodes, max_steps)
    env.close()
