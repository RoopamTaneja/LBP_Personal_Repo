import numpy as np
import random


class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize the Q-Learning agent.

        :param state_space: Number of states in the environment
        :param action_space: Number of actions in the environment
        :param alpha: Learning rate (0 to 1)
        :param gamma: Discount factor (0 to 1)
        :param epsilon: Exploration rate (0 to 1)
        """
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.state_space = state_space  # Number of states
        self.action_space = action_space  # Number of actions

        # Initialize Q-table with zeros
        self.Q = np.zeros((state_space[0], state_space[1], action_space))

    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy policy.

        :param state: Current state
        :return: Chosen action
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore: Random action
            return random.randint(0, self.action_space - 1)
        else:
            # Exploit: Choose action with highest Q-value
            return np.argmax(self.Q[state[0], state[1]])

    def update(self, state, action, reward, next_state):
        """
        Update the Q-value using the Q-learning update rule.

        :param state: Current state
        :param action: Action taken from current state
        :param reward: Reward received after taking action
        :param next_state: Next state after taking action
        """
        best_next_action = np.argmax(
            self.Q[next_state[0], next_state[1]]
        )  # Best action from next state
        # Q-learning update rule
        self.Q[state[0], state[1], action] = self.Q[
            state[0], state[1], action
        ] + self.alpha * (
            reward
            + self.gamma * self.Q[next_state[0], next_state[1], best_next_action]
            - self.Q[state[0], state[1], action]
        )


# Example Environment Setup (Gridworld)
class GridWorld:
    def __init__(self, grid_size, goal_state):
        """
        Simple GridWorld environment

        :param grid_size: Size of the grid (n x m)
        :param goal_state: Position of the goal state
        """
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.state = (0, 0)  # Start state

    def reset(self):
        """
        Reset the environment to the initial state
        """
        self.state = (0, 0)
        return self.state

    def step(self, action):
        """
        Take an action and return the next state and reward.

        :param action: Action to take (0 = up, 1 = down, 2 = left, 3 = right)
        :return: (next_state, reward)
        """
        row, col = self.state
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.grid_size[0] - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.grid_size[1] - 1, col + 1)

        self.state = (row, col)

        # Reward: 1 for reaching goal, 0 otherwise
        reward = 1 if self.state == self.goal_state else 0

        return self.state, reward


def train_q_learning(agent, env, episodes):
    """
    Train the Q-learning agent.

    :param agent: QLearningAgent
    :param env: Environment (GridWorld)
    :param episodes: Number of episodes to train for
    """
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if state == env.goal_state:
                done = True


# Initialize environment and agent
grid_dim = (5, 5)
env = GridWorld(grid_size=grid_dim, goal_state=(4, 4))
agent = QLearningAgent(
    state_space=grid_dim, action_space=4, alpha=0.1, gamma=0.9, epsilon=0.1
)

# Train the agent for 1000 episodes
train_q_learning(agent, env, episodes=1000)

# Display the learned Q-values
print("Q-table after training:")
for i in range(grid_dim[0]):
    for j in range(grid_dim[1]):
        print("{", end="")
        for a in range(agent.action_space):
            print(round(agent.Q[i, j, a], 2), end=" ")
        print("}", end="\t")
    print("")
