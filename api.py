import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ai_agent.q_learning import QLearningAgent
from src.main import init_environment, load_or_create_agent

# === CONFIG CORS ===
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === INIT ENV ET AGENT ===
env = init_environment()
agent, _ = load_or_create_agent("agent_model.npy", env)


# === MODELS ===
class MoveRequest(BaseModel):
    action: str  # "up", "down", "left", "right"


class MazeConfigRequest(BaseModel):
    width: int = 49
    height: int = 49
    seed: Optional[int] = 42


class TrainRequest(BaseModel):
    episodes: int = 100
    max_steps: int = 200
    model_file: str = "agent_model.npy"


# === ROUTES ===
@app.get("/maze")
def get_maze():
    return {
        "maze": env.maze.tolist(),
        "agent": list(env.state),
        "width": env.maze.shape[1],
        "height": env.maze.shape[0],
    }


@app.post("/ai-step")
def ai_move():

    current_state = tuple(env.state)

    old_epsilon = agent.epsilon
    agent.epsilon = 0
    action_idx = agent.choose_action(current_state)
    agent.epsilon = old_epsilon

    next_state, reward, done, info = env.step(action_idx)

    action_map_inv = {0: "up", 1: "down", 2: "left", 3: "right"}

    return {
        "agent": list(env.state),
        "action": action_map_inv[action_idx],
        "done": done,
        "reward": reward,
        "steps": env.n_steps,  # Ajout du compteur de pas
    }

    current_state = tuple(env.state)

    old_epsilon = agent.epsilon
    agent.epsilon = 0
    action_idx = agent.choose_action(current_state)
    agent.epsilon = old_epsilon

    next_state, reward, done, info = env.step(action_idx)

    action_map_inv = {0: "up", 1: "down", 2: "left", 3: "right"}

    return {
        "agent": list(env.state),
        "action": action_map_inv[action_idx],
        "done": done,
        "reward": reward,
        "steps": env.n_steps,  # Ajout du compteur de pas
    }


@app.post("/reset")
def reset_agent():
    env.reset()
    return {"agent": list(env.state), "steps": 0}


@app.post("/configure")
def configure_maze(config: MazeConfigRequest):
    global env, agent
    env = init_environment(width=config.width, height=config.height, seed=config.seed)
    agent = QLearningAgent(env)
    env.reset()
    return {
        "maze": env.maze.tolist(),
        "agent": list(env.state),
        "width": env.maze.shape[1],
        "height": env.maze.shape[0],
        "steps": 0,
    }


@app.delete("/memory")
def delete_memory():
    global agent
    model_file = "agent_model.npy"
    if os.path.exists(model_file):
        try:
            os.remove(model_file)
        except OSError:
            pass
    agent = QLearningAgent(env)
    env.reset()
    return {
        "agent": list(env.state),
        "steps": 0,
        "episodes_trained": agent.episodes_trained,
        "epsilon": agent.epsilon,
    }


@app.post("/train")
def train_agent(payload: TrainRequest):
    global env, agent
    agent.train(
        env,
        episodes=payload.episodes,
        max_steps=payload.max_steps,
        model_filename=payload.model_file,
    )
    env.reset()
    return {
        "episodes_trained": agent.episodes_trained,
        "epsilon": agent.epsilon,
        "agent": list(env.state),
        "steps": 0,
    }
