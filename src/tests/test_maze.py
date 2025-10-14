import pytest
import numpy as np
from src.environment.maze import MazeEnv

REWARD_GOAL = 20.0
REWARD_WALL = -0.5
REWARD_MOVE = -0.2


@pytest.fixture
def setup_maze():
    """Fixture pour initialiser l'environnement du labyrinthe."""
    maze = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ]
    )
    start = (0, 0)
    goal = (4, 4)
    env = MazeEnv(maze, start, goal)
    return env


def test_reset(setup_maze):
    """Test de la méthode reset."""
    env = setup_maze
    initial_state = env.reset()
    assert (
        initial_state == env.start
    ), "L'état initial devrait correspondre au point de départ."


def test_move_up(setup_maze):
    """Test du mouvement vers le haut."""
    env = setup_maze
    env.start = (
        1,
        0,
    )  # Positionner l'agent en (1,0) pour tester le mouvement vers le haut
    env.reset()
    state, reward, done, info = env.step(0)  # action = 0 (up)
    assert state == (
        0,
        0,
    ), "L'agent ne peut pas aller au-delà des limites du labyrinthe."
    assert (
        reward == REWARD_MOVE
    ), "Le mouvement dans une case valide mais sans objectif devrait donner une petite pénalité."


def test_move_down(setup_maze):
    """Test du mouvement vers le bas."""
    env = setup_maze
    env.reset()
    state, reward, done, info = env.step(1)  # action = 1 (down)
    assert state == (1, 0), "L'agent devrait descendre d'une case."
    assert (
        reward == REWARD_MOVE
    ), "Le mouvement dans une case valide mais sans objectif devrait donner une petite pénalité."


def test_move_left(setup_maze):
    """Test du mouvement vers la gauche."""
    env = setup_maze
    env.start = (
        2,
        1,
    )  # Positionner l'agent en (2,1) pour tester le mouvement vers la gauche
    env.reset()
    state, reward, done, info = env.step(2)  # action = 2 (left)
    assert state == (2, 0), "L'agent devrait se déplacer vers la gauche."
    assert (
        reward == REWARD_MOVE
    ), "Le mouvement dans une case valide mais sans objectif devrait donner une petite pénalité."


def test_move_right(setup_maze):
    """Test du mouvement vers la droite."""
    env = setup_maze
    env.start = (
        2,
        0,
    )  # Positionner l'agent en (2,0) pour tester le mouvement vers la droite
    env.reset()
    state, reward, done, info = env.step(3)  # action = 3 (right)
    assert state == (2, 1), "L'agent devrait se déplacer vers la droite."
    assert (
        reward == REWARD_MOVE
    ), "Le mouvement dans une case valide mais sans objectif devrait donner une petite pénalité."


def test_hit_wall(setup_maze):
    """Test des mouvements dans un mur."""
    env = setup_maze
    env.reset()
    state, reward, done, info = env.step(2)  # action = 2 (left) (mouvement vers un mur)
    assert state == (0, 0), "L'agent ne peut pas se déplacer dans un mur."
    assert (
        reward == REWARD_WALL
    ), "L'agent devrait recevoir une pénalité lorsqu'il frappe un mur."


def test_reach_goal(setup_maze):
    """Test lorsque l'agent atteint le but."""
    maze = np.array([[0, 1], [0, 1]])
    start = (0, 0)
    goal = (1, 0)
    env = MazeEnv(maze, start, goal)
    env.reset()
    # Avancer jusqu'au but (bas, bas, droite, droite)
    state, reward, done, info = env.step(1)  # right (atteindre l'objectif)

    assert done is True, "L'agent devrait avoir atteint l'objectif."
    assert state == env.goal, "L'agent devrait atteindre la position de l'objectif."
    assert (
        reward == REWARD_GOAL
    ), "L'agent devrait recevoir une grande récompense pour avoir atteint l'objectif."


def test_step_invalid_move(setup_maze):
    """Test pour vérifier que l'agent ne bouge pas lorsqu'il rencontre un mur."""
    env = setup_maze
    env.reset()
    old_state = env.state
    # Test si l'agent essaie de se déplacer en dehors des limites du labyrinthe
    state, reward, done, info = env.step(0)  # up (mouvement invalide vers un mur)
    assert (
        state == old_state
    ), "L'agent ne devrait pas pouvoir se déplacer dans une zone bloquée."
    assert (
        reward == REWARD_WALL
    ), "L'agent devrait être pénalisé s'il essaie de se déplacer dans un mur."


def test_render(setup_maze):
    """Test de la méthode render."""
    env = setup_maze
    env.reset()
    try:
        env.render()  # Juste vérifier que cela ne lève pas d'exception
    except Exception as e:
        pytest.fail(f"La méthode render a levé une exception: {e}")
