import random
from src.environment.maze import MazeEnv, generate_maze
from src.ai_agent.q_learning import QLearningAgent


def init_environment(width=23, height=23, seed=42):
    """Creates a reproducible MazeEnv environment."""
    random.seed(seed)
    maze = generate_maze(width, height)
    random.seed()  # reset global RNG
    return MazeEnv(maze, start=(1, 1), goal=(height - 2, width - 2))


def load_or_create_agent(model_file, env):
    """Loads an existing agent or creates a new one."""
    agent, saved_data = QLearningAgent.load(model_file, env=env)
    if agent is not None:
        return agent, True
    agent = QLearningAgent(env)
    print("\n🆕 New model created")
    return agent, False


def main(
    auto_train=False,
    episodes=5000,
    max_steps=500,
    model_file="agent_model.npy",
    reset=False,
):
    """Starts agent training and/or testing.

    - auto_train=True  → no input, everything runs automatically.
    - reset=True       → ignores any existing model.
    """

    env = init_environment()
    agent, loaded = load_or_create_agent(model_file, env)

    if auto_train:
        print("\n🤖 AUTOMATIC MODE ENABLED")
        if reset or not loaded:
            print("➡️  Creating a new agent (no data loaded)")
            agent = QLearningAgent(env)
        else:
            print("➡️  Existing model detected — additional training")
            agent.reset_exploration(epsilon=0.3)
        agent.train(
            env, episodes=episodes, max_steps=max_steps, model_filename=model_file
        )

    else:
        # === INTERACTIVE MODE ===
        if loaded:
            print("\n🤔 Options:")
            print("  1. Continue training")
            print("  2. Test the current model")
            print("  3. Reset (memory loss)")
            choice = input("\nYour choice (1/2/3): ").strip()

            if choice == "1":
                nb = int(input("Number of additional episodes: "))
                agent.reset_exploration(epsilon=0.3)
                agent.train(
                    env, episodes=nb, max_steps=max_steps, model_filename=model_file
                )
            elif choice == "3":
                print("\n⚠️  Complete reset...")
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
    print("🎯 FINAL TEST")
    print("=" * 60)
    agent.test(env, max_steps=max_steps, render=False)


if __name__ == "__main__":
    # ⚙️ Manual mode by default
    main(auto_train=False, reset=False, episodes=500)
