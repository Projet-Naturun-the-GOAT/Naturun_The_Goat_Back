import numpy as np
import random
import os


class QLearningAgent:
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = 4  # up, down, left, right
        self.q_table = {}  # {(row, col): [Q_up, Q_down, Q_left, Q_right]}
        self.episodes_trained = 0  # Total episode counter

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
            "q_table": self.q_table,
            "best_reward": best_reward,
            "episodes_trained": self.episodes_trained,
            "hyperparameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
        }

        # Create folder if necessary
        os.makedirs(
            os.path.dirname(filename) if os.path.dirname(filename) else ".",
            exist_ok=True,
        )

        with open(filename, "wb") as f:
            np.save(f, save_data, allow_pickle=True)
        print(f"💾 Modèle sauvegardé: {filename}")
        print(
            f"   └─ Episodes: {self.episodes_trained}, Epsilon: {self.epsilon:.4f}, Best reward: {best_reward}"
        )

    @classmethod
    def load(cls, filename, env=None):
        """Charge un agent avec TOUTE sa mémoire"""
        try:
            with open(filename, "rb") as f:
                saved_data = np.load(f, allow_pickle=True).item()

            agent = cls(env=env)

            # Restore the Q-table (THE MEMORY!)
            agent.q_table = saved_data.get("q_table", {})

            # Restore the number of episodes
            agent.episodes_trained = saved_data.get("episodes_trained", 0)

            # Restore the hyperparameters
            if "hyperparameters" in saved_data:
                params = saved_data["hyperparameters"]
                agent.alpha = params.get("alpha", agent.alpha)
                agent.gamma = params.get("gamma", agent.gamma)
                agent.epsilon = params.get("epsilon", agent.epsilon)
                agent.epsilon_min = params.get("epsilon_min", agent.epsilon_min)
                agent.epsilon_decay = params.get("epsilon_decay", agent.epsilon_decay)

            best_reward = saved_data.get("best_reward", "Inconnu")

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

    def train(
        self,
        env,
        episodes=50,
        max_steps=200,
        save_interval=100,
        model_filename="agent_model.npy",
    ):
        """Entraînement avec sauvegarde automatique de la mémoire"""

        # Load the previous best reward
        best_reward = float("-inf")
        if os.path.exists(model_filename):
            try:
                with open(model_filename, "rb") as f:
                    saved_data = np.load(f, allow_pickle=True).item()
                    best_reward = saved_data.get("best_reward", float("-inf"))
            except (EOFError, ValueError, FileNotFoundError):
                print(f"⚠️  Fichier de modèle invalide ou corrompu : {model_filename}")

        print(f"\n{'='*60}")
        print("🎓 DÉBUT DE L'ENTRAÎNEMENT")
        print(f"{'='*60}")
        print(f"Episodes prévus: {episodes}")
        print(f"Episodes déjà effectués: {self.episodes_trained}")
        print(f"Epsilon initial: {self.epsilon:.4f}")
        print(f"Meilleur reward actuel: {best_reward}")
        print(f"{'='*60}\n")

        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0

            # EPISODE EXPLORATION
            while not done and steps < max_steps:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

            # POST-EPISODE UPDATE
            self.episodes_trained += 1
            self.decay_epsilon()  # Reduce exploration progressively

            # Periodic display
            if ep % 10 == 0 or ep == episodes - 1:
                print(
                    f"Episode {ep:4d}/{episodes} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | Epsilon: {self.epsilon:.4f} | États connus: {len(self.q_table)}"
                )

            # SIGNIFICANT IMPROVEMENT
            if best_reward < 0:
                improvement_threshold = -0.05
            else:
                improvement_threshold = 0.05

            if total_reward > best_reward * (1 + improvement_threshold):
                best_reward = total_reward

            # PERIODIC CHECKPOINT (every N iterations)
            if (ep + 1) % save_interval == 0:
                checkpoint_file = f"checkpoints/agent_ep{self.episodes_trained}.npy"
                self.save(checkpoint_file, best_reward=best_reward)

        # FINAL SAVE
        print(f"\n{'='*60}")
        print("💾 Sauvegarde finale...")
        self.save(model_filename, best_reward=best_reward)
        print(f"{'='*60}")

    def test(self, env, max_steps=200, render=True):
        """Test de l'agent (MODE EXPLOITATION PURE)"""
        # Save current epsilon
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # Disable exploration for testing

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
            self.ensure_state(state)
            action = int(np.argmax(self.q_table[state]))  # Always the best action
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

        # Restore epsilon
        self.epsilon = original_epsilon

        return total_reward, steps
