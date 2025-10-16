from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#Autorise ton front (ex: React/Vite sur localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/bonjour")
def dire_bonjour():
    return {"message": "Bonjour 👋 depuis l'API FastAPI !"}

#Pour exécution directe (ex: python api.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
