# can check this : https://medium.com/@qempsil0914/deep-q-learning-part2-double-deep-q-network-double-dqn-b8fc9212bbb2

import numpy as np
import tensorflow as tf
from collections import deque
import random

class DDQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=32, tau=0.125):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        
        # Experience replay buffer
        self.memory = deque(maxlen=2000)
        
        # Q-Network and Target Network
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def update_target_model(self):
        # Update target model with weights of the model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_action = np.argmax(self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
                target = reward + self.gamma * self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0][next_action]
            
            target_f = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            target_f[0][action] = target
            
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_target_model()

# Usage example
if __name__ == "__main__":
    state_size = 4  # Example state size (e.g., CartPole)
    action_size = 2  # Example action size (e.g., Left and Right)

    agent = DDQNAgent(state_size, action_size)

    # Assume we have an environment to interact with
    for episode in range(1000):  # Example for 1000 episodes
        state = np.random.rand(state_size)  # Random initial state for illustration
        done = False
        while not done:
            action = agent.act(state)
            next_state = np.random.rand(state_size)  # Random next state for illustration
            reward = 1  # Dummy reward
            done = random.random() < 0.1  # Randomly end the episode

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
        print(f"Episode {episode} completed")

'''
Explanation:
DDQNAgent class: The agent is designed with basic attributes and methods, such as:

act(): Chooses an action based on epsilon-greedy policy.
remember(): Stores experiences in memory.
replay(): Samples from memory to train the model using Double DQN.
update_target_model(): Synchronizes the target model with the main model.
Neural Network (Q-Network): The model consists of two hidden layers of 24 units each with ReLU activations, and the output layer has one unit per action. The network is trained using Mean Squared Error loss.

Training: In replay(), the target is computed using the Double DQN mechanism. The main model's Q-values are updated by using the next action from the main model and the Q-values from the target model.

Epsilon Decay: Epsilon is decayed after each episode to reduce exploration over time.
'''
