# src/ai_agent/q_learning.py
import numpy as np
import random
import pickle
from src.python.environment.maze import MazeEnv

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

    # === Persistance ===
    def save(self, filepath="q_table.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath="q_table.pkl"):
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)
            

def generate_maze(width=31, height=31):
    # largeur et hauteur impaires pour des murs bien définis
    maze = np.ones((height, width), dtype=int)

    def carve(r, c):
        dirs = [(2,0), (-2,0), (0,2), (0,-2)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < height-1 and 1 <= nc < width-1 and maze[nr, nc] == 1:
                maze[nr-dr//2, nc-dc//2] = 0
                maze[nr, nc] = 0
                carve(nr, nc)

    # point de départ
    maze[1,1] = 0
    carve(1,1)

    return maze


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

        if ep % 100 == 0 or ep == episodes - 1:
            print(f"Episode {ep}, total_reward={total_reward}, steps={steps}")


def test(agent, env, max_steps=2000):
    state = env.reset()
    env.render()
    total_reward = 0
    print(f"---------")
    for step in range(max_steps):
        agent.ensure_state(state)
        action = int(np.argmax(agent.q_table[state]))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        total_reward += reward
        if done:
            print(f"Sortie atteinte en {step+1} étapes avec reward {total_reward}")
            break
        
if __name__ == "__main__":
    # maze = np.array([
    #     [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #     [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    #     [1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1],
    #     [1,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1],
    #     [1,0,1,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,0,1],
    #     [1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1],
    #     [1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1],
    #     [1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1],
    #     [1,0,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1],
    #     [1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1],
    #     [1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1],
    #     [1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
    #     [1,0,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1],
    #     [1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1],
    #     [1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1],
    #     [1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0],
    #     [1,0,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,0],
    #     [1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0],
    #     [1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0],
    #     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # ], dtype=int)

    random.seed(42)   # fixe la seed du module random
    np.random.seed(42)  # fixe la seed de numpy
    maze = generate_maze(51, 51)

    env = MazeEnv(maze, start=(1, 1), goal=(49, 49))
    agent = QLearningAgent(env)
    agent.load("q_table.pkl")

    print("=== Entraînement ===")
    train(agent, env, episodes=10000, max_steps=2000)
    
    agent.save("q_table.pkl")

    print("\n=== Test ===")
    test(agent, env)