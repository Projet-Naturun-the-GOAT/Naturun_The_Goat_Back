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
        "maze": env.maze.tolist(), 
        "agent": list(env.state),  
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
        "steps": env.n_steps  # Ajout du compteur de pas
    }

@app.post("/reset")
def reset_agent():
    env.reset() 
    return {
        "agent": list(env.state),
        "steps": 0
    }
