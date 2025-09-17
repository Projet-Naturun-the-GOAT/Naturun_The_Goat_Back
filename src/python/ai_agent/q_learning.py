# src/ai_agent/q_learning.py
import numpy as np
import random
from src.environment.maze import MazeEnv

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = 4  # up, down, left, right
        self.q_table = {}  # {(row, col): [Q_up, Q_down, Q_left, Q_right]}

    def ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

    def choose_action(self, state):
        self.ensure_state(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        self.ensure_state(state)
        self.ensure_state(next_state)

        q_predict = self.q_table[state][action]
        q_target = reward
        if not done:
            q_target += self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (q_target - q_predict)


def train(agent, env, episodes=50, max_steps=200):
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        if ep % 10 == 0 or ep == episodes - 1:
            print(f"Episode {ep}, total_reward={total_reward}, steps={steps}")


def test(agent, env, max_steps=200):
    state = env.reset()
    env.render()
    print(f"---------")
    for step in range(max_steps):
        agent.ensure_state(state)
        action = int(np.argmax(agent.q_table[state]))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        if done:
            print(f"Sortie atteinte en {step+1} étapes avec reward {reward}")
            break
if __name__ == "__main__":
    maze = np.array([
        [0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0],
        [1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0],
        [0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0],
        [0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0],
        [0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0],
        [0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1],
        [0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,1,0,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0],
        [0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
        [1,1,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,0],
        [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
        [0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,1,1,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    ], dtype=int)

    env = MazeEnv(maze, start=(0, 0), goal=(19, 19))
    agent = QLearningAgent(env)

    print("=== Entraînement ===")
    train(agent, env, episodes=10000, max_steps=44)

    print("\n=== Test ===")
    test(agent, env)