from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from environment.maze import generate_maze

app = FastAPI()

# Allow requests from front-end applications (e.g., React, Vite, Express) by configuring CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/bonjour")
def dire_bonjour():
    return {"message": "Bonjour 👋 depuis l'API FastAPI !"}


# ---------- ASCII endpoint (Option A) ----------
def make_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def grid_to_ascii(grid, start=None, end=None) -> str:
    """grid: liste de listes 0/1 ; 1 = mur '#', 0 = chemin '.'"""
    H, W = len(grid), len(grid[0])
    lines = []
    for r in range(H):
        row_chars = []
        for c in range(W):
            if start and (r, c) == start:
                row_chars.append("S")
            elif end and (r, c) == end:
                row_chars.append("E")
            else:
                row_chars.append("#" if grid[r][c] == 1 else ".")
        lines.append("".join(row_chars))
    return "\n".join(lines)


@app.get("/api/maze/ascii", response_class=PlainTextResponse)
def api_maze_ascii(
    width: int = Query(31, ge=5),
    height: int = Query(31, ge=5),
    start_row: int | None = None,
    start_col: int | None = None,
    end_row: int | None = None,
    end_col: int | None = None,
):
    # dimensions impaires pour l'algorithme
    W, H = make_odd(width), make_odd(height)

    # génère la grille (numpy array ou list selon ton implémentation)
    grid = generate_maze(W, H)
    try:
        grid = grid.tolist()  # si c'est un np.array
    except AttributeError:
        pass  # déjà une liste

    # positions S/E par défaut si non fournies
    start = (
        start_row if start_row is not None else 1,
        start_col if start_col is not None else 1,
    )
    end = (
        end_row if end_row is not None else H - 2,
        end_col if end_col is not None else W - 2,
    )

    ascii_map = grid_to_ascii(grid, start=start, end=end)
    return ascii_map


# -----------------------------------------------

# Pour exécution directe (ex: python api.py)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
