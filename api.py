from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


# === ROUTES ===
@app.get("/maze")
def get_maze():
    return {
        "maze": env.maze.tolist(),  # convertir numpy array en liste
        "agent": list(env.state),  # convertir tuple en liste pour JSON
    }


@app.post("/step")
def move_agent(req: MoveRequest):
    action_map = {"up": 0, "down": 1, "left": 2, "right": 3}
    if req.action not in action_map:
        return {"error": "Invalid action"}

    next_state, reward, done, info = env.step(action_map[req.action])
    return {"agent": list(env.state), "reward": reward, "done": done}


@app.post("/reset")
def reset_agent():
    env.reset()
    return {"agent": list(env.state)}
