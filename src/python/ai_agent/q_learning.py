# src/ai_agent/q_learning.py
import numpy as np
import random
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
    
    def save(self, filename, best_reward=None):
        """Sauvegarde de la Q-table dans le fichier agent_model.json"""
        save_agent = {
            'q_table': self.q_table,
            'best_reward': best_reward,
            "hyperparameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon
            }
        }
        with open(filename, 'wb') as f:
            np.save(f, save_agent, allow_pickle=True)
            print(f"Modèle sauvegardé dans {filename}")
    
    @classmethod        
    def load(self, filename):
        try:
            with open(filename, 'rb') as f:
                saved_data = np.load(f, allow_pickle=True).item()
                
                agent = cls(env=None)  
                agent.q_table = saved_data['q_table']
                
                if 'hyperparameters' in saved_data:
                    params = saved_data['hyperparameters']
                    agent.alpha = params.get('alpha', agent.alpha)
                    agent.gamma = params.get('gamma', agent.gamma)
                    agent.epsilon = params.get('epsilon', agent.epsilon)
                    
                best_reward = saved_data.get('best_reward', 'Inconnu')
                print(f"✓ Modèle chargé depuis {filename}")
                print(f"  - Meilleur reward: {best_reward}")
                return agent, saved_data
        except FileNotFoundError:
            print(f"✗ Modèle non trouvé à {filename}")
            return None, None
        

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
    filename = "agent_model.npy"
    state = env.reset()
    env.render()
    print(f"---------")
    try:
        with open(filename, "rb") as f:
            saved_data = np.load(f, allow_pickle=True).item()
            previous_best_reward = saved_data.get('best_reward', float('-inf'))
            print(f"Dernier modèle entraîné, chargé avec un meilleur reward de {previous_best_reward}")
    except (FileNotFoundError, EOFError):
        previous_best_reward = float('-inf')
        print("Aucun modèle pré-entraîné trouvé, démarrage d'un nouvel entraînement")
    for step in range(max_steps):
        agent.ensure_state(state)
        action = int(np.argmax(agent.q_table[state]))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        if done:
            print(f"Sortie atteinte en {step+1} étapes avec reward {reward}")
            if reward > previous_best_reward:
                print(f"Nouveau meilleur reward! Ancien: {previous_best_reward}, Nouveau: {reward}")
                agent.save(filename, best_reward=reward) 
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
    train(agent, env, episodes=10000, max_steps=45)

    print("\n=== Test ===")
    test(agent, env)