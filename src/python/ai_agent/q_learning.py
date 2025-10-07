# src/ai_agent/q_learning.py
import numpy as np
import random
import os
from src.python.environment.maze import MazeEnv


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = 4  # up, down, left, right
        self.q_table = {}  # {(row, col): [Q_up, Q_down, Q_left, Q_right]}
        self.episodes_trained = 0  # Compteur d'épisodes total

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

    def decay_epsilon(self):
        """Réduit epsilon progressivement"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_exploration(self, epsilon=0.3):
        """Réactive l'exploration si l'agent est bloqué"""
        old_epsilon = self.epsilon
        self.epsilon = epsilon
        print(f"🔄 Exploration réactivée: {old_epsilon:.4f} → {epsilon:.4f}")

    def save(self, filename, best_reward=None):
        """Sauvegarde complète de l'agent avec toutes ses informations"""
        save_data = {
            'q_table': self.q_table,
            'best_reward': best_reward,
            'episodes_trained': self.episodes_trained,
            'hyperparameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            }
        }
        
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'wb') as f:
            np.save(f, save_data, allow_pickle=True)
        print(f"💾 Modèle sauvegardé: {filename}")
        print(f"   └─ Episodes: {self.episodes_trained}, Epsilon: {self.epsilon:.4f}, Best reward: {best_reward}")
    
    @classmethod        
    def load(cls, filename, env=None):
        """Charge un agent avec TOUTE sa mémoire"""
        try:
            with open(filename, 'rb') as f:
                saved_data = np.load(f, allow_pickle=True).item()
                
            agent = cls(env=env)
            
            # Restaurer la Q-table (LA MÉMOIRE!)
            agent.q_table = saved_data.get('q_table', {})
            
            # Restaurer le nombre d'épisodes
            agent.episodes_trained = saved_data.get('episodes_trained', 0)
            
            # Restaurer les hyperparamètres
            if 'hyperparameters' in saved_data:
                params = saved_data['hyperparameters']
                agent.alpha = params.get('alpha', agent.alpha)
                agent.gamma = params.get('gamma', agent.gamma)
                agent.epsilon = params.get('epsilon', agent.epsilon)
                agent.epsilon_min = params.get('epsilon_min', agent.epsilon_min)
                agent.epsilon_decay = params.get('epsilon_decay', agent.epsilon_decay)
            
            best_reward = saved_data.get('best_reward', 'Inconnu')
            
            print(f"✅ Modèle chargé: {filename}")
            print(f"   ├─ Q-table: {len(agent.q_table)} états connus")
            print(f"   ├─ Episodes entraînés: {agent.episodes_trained}")
            print(f"   ├─ Epsilon actuel: {agent.epsilon:.4f}")
            print(f"   └─ Meilleur reward: {best_reward}")
            
            return agent, saved_data
            
        except FileNotFoundError:
            print(f"⚠️  Aucun modèle trouvé: {filename}")
            return None, None
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return None, None
        

def train(agent, env, episodes=50, max_steps=200, save_interval=100, model_filename="agent_model.npy"):
    """Entraînement avec sauvegarde automatique de la mémoire"""
    
    # Charger le meilleur reward précédent
    best_reward = float('-inf')
    if os.path.exists(model_filename):
        try:
            with open(model_filename, "rb") as f:
                saved_data = np.load(f, allow_pickle=True).item()
                best_reward = saved_data.get('best_reward', float('-inf'))
        except:
            pass

    print(f"\n{'='*60}")
    print(f"🎓 DÉBUT DE L'ENTRAÎNEMENT")
    print(f"{'='*60}")
    print(f"Episodes prévus: {episodes}")
    print(f"Episodes déjà effectués: {agent.episodes_trained}")
    print(f"Epsilon initial: {agent.epsilon:.4f}")
    print(f"Meilleur reward actuel: {best_reward}")
    print(f"{'='*60}\n")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # EXPLORATION DE L'ÉPISODE
        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        # MISE À JOUR POST-ÉPISODE
        agent.episodes_trained += 1
        agent.decay_epsilon()  # Réduire l'exploration progressivement

        # Affichage périodique
        if ep % 10 == 0 or ep == episodes - 1:
            print(f"Episode {ep:4d}/{episodes} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | Epsilon: {agent.epsilon:.4f} | États connus: {len(agent.q_table)}")
        
        # 🔥 SAUVEGARDE SI NOUVEAU RECORD
        if total_reward > best_reward:
            best_reward = total_reward
            print(f"   🎉 NOUVEAU RECORD! {total_reward:.2f} → Sauvegarde automatique")
            agent.save(model_filename, best_reward=best_reward)
        
        # 💾 CHECKPOINT PÉRIODIQUE (toutes les N itérations)
        if (ep + 1) % save_interval == 0:
            checkpoint_file = f"checkpoints/agent_ep{agent.episodes_trained}.npy"
            agent.save(checkpoint_file, best_reward=best_reward)
    
    # SAUVEGARDE FINALE
    print(f"\n{'='*60}")
    print("💾 Sauvegarde finale...")
    agent.save(model_filename, best_reward=best_reward)
    print(f"{'='*60}")


def test(agent, env, max_steps=200, render=True):
    """Test de l'agent (MODE EXPLOITATION PURE)"""
    # Sauvegarder l'epsilon actuel
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Désactiver l'exploration pour le test
    
    state = env.reset()
    total_reward = 0
    steps = 0
    
    if render:
        print(f"\n{'='*60}")
        print("🧪 TEST DE L'AGENT (pas d'exploration)")
        print(f"{'='*60}")
        env.render()
        print("-" * 40)
    
    for step in range(max_steps):
        agent.ensure_state(state)
        action = int(np.argmax(agent.q_table[state]))  # Toujours la meilleure action
        next_state, reward, done, _ = env.step(action)
        
        total_reward += reward
        state = next_state
        steps = step + 1
        
        if render:
            env.render()
        
        if done:
            if reward > 0:
                print(f"\n✅ SUCCÈS en {steps} étapes!")
            else:
                print(f"\n❌ ÉCHEC après {steps} étapes")
            print(f"   Reward total: {total_reward:.2f}")
            break
    else:
        print(f"\n⏱️  Timeout ({max_steps} étapes max)")
        print(f"   Reward total: {total_reward:.2f}")
    
    # Restaurer epsilon
    agent.epsilon = original_epsilon
    
    return total_reward, steps


def generate_maze(width=31, height=31):
    """Génère un labyrinthe aléatoire avec l'algorithme de backtracking récursif"""
    maze = np.ones((height, width), dtype=int)

    def carve(r, c):
        random.seed(42)
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < height-1 and 1 <= nc < width-1 and maze[nr, nc] == 1:
                maze[nr-dr//2, nc-dc//2] = 0
                maze[nr, nc] = 0
                carve(nr, nc)

    maze[1, 1] = 0
    carve(1, 1)
    random.seed()  # Réinitialiser le seed global
    return maze


if __name__ == "__main__":
    # Générer ou charger un labyrinthe
    maze = generate_maze(21, 21)  # Plus petit pour un apprentissage plus rapide
    env = MazeEnv(maze, start=(1, 1), goal=(19, 19))

    model_file = "agent_model.npy"

    # 1️⃣ TENTATIVE DE CHARGEMENT D'UN MODÈLE EXISTANT
    loaded_agent, saved_data = QLearningAgent.load(model_file, env=env)

    if loaded_agent is not None:
        # Modèle trouvé : proposer de continuer ou tester
        print("\n🤔 Options:")
        print("  1. Continuer l'entraînement (l'agent garde sa mémoire)")
        print("  2. Tester le modèle actuel")
        print("  3. Recommencer de zéro (PERTE DE LA MÉMOIRE)")

        choice = input("\nVotre choix (1/2/3): ").strip()

        if choice == "1":
            # CONTINUER L'APPRENTISSAGE
            nb_episodes = int(input("Nombre d'épisodes supplémentaires: "))
            loaded_agent.reset_exploration(epsilon=0.3)  # Réactiver un peu l'exploration
            train(loaded_agent, env, episodes=nb_episodes, max_steps=500, model_filename=model_file)
            agent = loaded_agent

        elif choice == "2":
            # JUSTE TESTER
            agent = loaded_agent

        else:
            # RECOMMENCER
            print("\n⚠️  Vous allez perdre toute la mémoire de l'agent!")
            confirm = input("Confirmer (oui/non): ").strip().lower()
            if confirm == "oui":
                agent = QLearningAgent(env)
                train(agent, env, episodes=5000, max_steps=500, model_filename=model_file)
            else:
                print("Annulé.")
                exit()
    else:
        # 2️⃣ PAS DE MODÈLE : CRÉER UN NOUVEL AGENT
        print("\n🆕 Création d'un nouvel agent")
        agent = QLearningAgent(env)
        train(agent, env, episodes=5000, max_steps=500, model_filename=model_file)

    # 3️⃣ TEST FINAL
    print("\n" + "="*60)
    print("🎯 TEST FINAL")
    print("="*60)
    test(agent, env, max_steps=500, render=True)
