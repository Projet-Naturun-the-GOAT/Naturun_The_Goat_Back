import random
import os
from src.environment.maze import MazeEnv, generate_maze
from src.ai_agent.q_learning import QLearningAgent


def init_environment(width=23, height=23, seed=42, level=1):
    """Crée un environnement MazeEnv reproductible."""
    random.seed(seed)
    maze = generate_maze(width, height)
    random.seed()  # reset global RNG
    return MazeEnv(maze, start=(1, 1), goal=(height - 2, width - 2), level=level)


def load_or_create_agent(model_file, env):
    """Charge un agent existant ou en crée un nouveau."""
    agent, saved_data = QLearningAgent.load(model_file, env=env)
    if agent is not None:
        return agent, True
    agent = QLearningAgent(env)
    print("\n🆕 Nouveau modèle créé")
    return agent, False


def main(
    auto_train=False,
    episodes=5000,
    max_steps=400,
    model_file="agent_model.npy",
    reset=False,
):
    """Lance l'entraînement et/ou test de l'agent.

    - auto_train=True  → aucun input, tout enchaîne automatiquement.
    - reset=True       → ignore tout modèle existant.
    """
    print("\n🤖 MODE INTERACTIF ACTIVÉ")

    choice = input("\nVeuillez choisir le niveau du labyrinthe (1-2): ").strip()

    env = init_environment(level=int(choice))
    agent, loaded = load_or_create_agent(model_file, env)

    if auto_train:
        print("\n🤖 MODE AUTOMATIQUE ACTIVÉ")
        if reset or not loaded:
            print("➡️  Création d’un nouvel agent (aucune donnée chargée)")
            agent = QLearningAgent(env)
        else:
            print("➡️  Modèle existant détecté — entraînement supplémentaire")
            agent.reset_exploration(epsilon=0.3)
        agent.train(
            env, episodes=episodes, max_steps=max_steps, model_filename=model_file
        )

    else:
        # === INTERACTIVE MODE ===
        if loaded:
            print("\n🤔 Options:")
            print("  1. Continuer l'entraînement")
            print("  2. Tester le modèle actuel")
            print("  3. Réinitialiser (perte de mémoire)")
            choice = input("\nVotre choix (1/2/3): ").strip()

            if choice == "1":
                nb = int(input("Nombre d'épisodes supplémentaires: "))
                agent.reset_exploration(epsilon=0.3)
                agent.train(
                    env, episodes=nb, max_steps=max_steps, model_filename=model_file
                )
            elif choice == "3":
                print("\n⚠️  Réinitialisation complète...")

                # Supprimer le modèle précédent s’il existe
                if os.path.exists(model_file):
                    try:
                        os.remove(model_file)
                        print(f"🗑️  Ancien modèle supprimé : {model_file}")
                    except OSError as e:
                        print(
                            f"⚠️  Échec de la suppression du modèle : {model_file} ({e})"
                        )
                else:
                    print("ℹ️  Aucun modèle existant à supprimer.")

                # 🔁 Recréation complète de l’agent
                env.reset()
                agent = QLearningAgent(env)
                agent.train(
                    env,
                    episodes=episodes,
                    max_steps=max_steps,
                    model_filename=model_file,
                )
        else:
            agent.train(
                env, episodes=episodes, max_steps=max_steps, model_filename=model_file
            )

    # === FINAL TEST ===
    print("\n" + "=" * 60)
    print("🎯 TEST FINAL")
    print("=" * 60)
    agent.test(env, max_steps=max_steps, render=True)


if __name__ == "__main__":
    # ⚙️ Manual mode by default
    main(auto_train=False, reset=False, episodes=500)
