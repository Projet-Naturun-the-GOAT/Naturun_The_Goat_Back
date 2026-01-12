import os
import random

import numpy as np


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
        self.episodes_trained = 0  # Compteur d'épisodes total
        self.best_rewards = {}

    def get_full_state(self, state):
        # Pour le niveau 2, inclure la clé dans l'état
        if hasattr(self.env, "level") and self.env.level == 2 and hasattr(self.env, "has_key"):
            return (state[0], state[1], self.env.has_key)
        return state
    
    def ensure_state(self, state):
        state = self.get_full_state(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
            
    def choose_action(self, state):
        state = self.get_full_state(state)
        self.ensure_state(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        state = self.get_full_state(state)
        next_state = self.get_full_state(next_state)
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
        if not os.path.exists(filename):
            print(f"⚠️  Aucun modèle trouvé: {filename}")
            return None, {}
        try:
            with open(filename, "rb") as f:
                saved_data = np.load(f, allow_pickle=True).item()

            agent = cls(env=env)

            # Restore the Q-table (THE MEMORY!)
            agent.q_table = saved_data.get("q_table", {})

            # Restaurer les meilleures récompenses
            agent.best_rewards = saved_data.get("best_rewards", {})

            # Restaurer le nombre d'épisodes
            agent.episodes_trained = saved_data.get("episodes_trained", 0)

            # Restore the hyperparameters
            if "hyperparameters" in saved_data:
                params = saved_data["hyperparameters"]
                agent.alpha = params.get("alpha", agent.alpha)
                agent.gamma = params.get("gamma", agent.gamma)
                agent.epsilon = params.get("epsilon", agent.epsilon)
                agent.epsilon_min = params.get(
                    "epsilon_min", agent.epsilon_min)
                agent.epsilon_decay = params.get(
                    "epsilon_decay", agent.epsilon_decay)

            saved_best_reward = saved_data.get("best_reward", float("-inf"))
            if isinstance(saved_best_reward, dict):
                agent.best_rewards = saved_best_reward
            else:
                agent.best_rewards = {getattr(env, "level", 1): saved_best_reward}

            print(f"✅ Modèle chargé: {filename}")
            print(f"   ├─ Q-table: {len(agent.q_table)} états connus")
            print(f"   ├─ Episodes entraînés: {agent.episodes_trained}")
            print(f"   ├─ Epsilon actuel: {agent.epsilon:.4f}")
            print(f"   └─ Meilleur reward: {"level :" + str(getattr(env, 'level', 1))}: {agent.best_rewards.get(getattr(env, 'level', 1), float('-inf'))}")

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
        model_filename="agent_model.npy",
    ):
        """Entraînement avec sauvegarde automatique de la mémoire"""

        # Charger le meilleur reward précédent
        reward_buffer = []
        buffer_size = 50
        level = getattr(env, "level", 1)
        best_reward = self.best_rewards.get(level, float("-inf"))
        if os.path.exists(model_filename):
            try:
                with open(model_filename, "rb") as f:
                    saved_data = np.load(f, allow_pickle=True).item()
                    saved_best_reward = saved_data.get("best_reward", float("-inf"))
                    if isinstance(saved_best_reward, dict):
                        best_reward = saved_best_reward.get(level, float("-inf"))
                    else:
                        best_reward = saved_best_reward
            except (EOFError, ValueError, FileNotFoundError):
                print(
                    f"⚠️  Fichier de modèle invalide ou corrompu : {model_filename}")

        print(f"\n{'='*60}")
        print("🎓 DÉBUT DE L'ENTRAÎNEMENT")
        print(f"{'='*60}")
        print(f"Episodes prévus: {episodes}")
        print(f"Episodes déjà effectués: {self.episodes_trained}")
        print(f"Epsilon initial: {self.epsilon:.4f}")
        print(f"Meilleur reward actuel: {best_reward}")
        print(f"{'='*60}\n")

        save_interval = 100
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

            # Ajout du reward total de l'épisode au buffer
            reward_buffer.append(total_reward)
            if len(reward_buffer) > buffer_size:
                reward_buffer.pop(0)

            # MISE À JOUR POST-ÉPISODE
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

            if done and total_reward > best_reward * (1 + improvement_threshold):
                best_reward = total_reward
                self.best_rewards[level] = best_reward

            # PERIODIC CHECKPOINT (every N iterations)
            if (ep + 1) % save_interval == 0:
                checkpoint_file = f"checkpoints/agent_ep{self.episodes_trained}.npy"
                self.save(checkpoint_file, best_reward=self.best_rewards)

        # FINAL SAVE
        print(f"\n{'='*60}")
        print("💾 Sauvegarde finale...")
        self.save(model_filename, best_reward=self.best_rewards)
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
            state_key = self.get_full_state(state)
            self.ensure_state(state_key)
            # Toujours la meilleure action
            action = int(np.argmax(self.q_table[state_key]))
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
