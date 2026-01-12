import numpy as np
import pytest

from src.environment.maze import MazeEnv, generate_maze

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
    )  # Position the agent at (1,0) to test upward movement
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
    )  # Position the agent at (2,1) to test leftward movement
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
    )  # Position the agent at (2,0) to test rightward movement
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
    state, reward, done, info = env.step(
        2
    )  # action = 2 (left) (movement towards a wall)
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
    # Move to the goal (down, down, right, right)
    state, reward, done, info = env.step(1)  # right (reach the objective)

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
    # Test if the agent tries to move outside the maze boundaries
    state, reward, done, info = env.step(0)  # up (invalid movement towards a wall)
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
        env.render()  # Just check that this doesn't raise an exception
    except Exception as e:
        pytest.fail(f"La méthode render a levé une exception: {e}")


def test_backtrack_penalty(setup_maze):
    """Test de la pénalité d'aller-retour immédiat."""
    env = setup_maze
    env.start = (2, 0)
    env.reset()
    env.step(3)  # right to (2,1)
    _, reward, _, _ = env.step(2)  # left back to (2,0)
    assert reward <= REWARD_MOVE - 1.0, "L'aller-retour devrait appliquer une pénalité."


def test_level_2_key_pickup():
    """Test de la récupération de la clé au niveau 2."""
    maze = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    start = (1, 2)
    goal = (2, 4)
    env = MazeEnv(maze, start, goal, level=2)
    env.reset()
    # key_position is (1, width-2) -> (1,3)
    state, reward, done, info = env.step(3)  # right to key
    assert state == (1, 3)
    assert reward == pytest.approx(15.0)
    assert env.has_key is True
    assert done is False


def test_level_2_door_without_key_blocks():
    """Test qu'une porte bloque sans clé au niveau 2."""
    maze = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    start = (0, 2)
    goal = (1, 1)
    env = MazeEnv(maze, start, goal, level=2)
    env.reset()
    # Door is at (1,2) around the goal
    state, reward, done, info = env.step(1)  # down into door
    assert state == start
    assert reward == pytest.approx(-5.0)
    assert env.has_key is False
    assert done is False


def test_level_2_door_with_key_opens():
    """Test qu'une porte s'ouvre avec la clé au niveau 2."""
    maze = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    start = (0, 2)
    goal = (1, 1)
    env = MazeEnv(maze, start, goal, level=2)
    env.reset()
    env.has_key = True
    state, reward, done, info = env.step(1)  # down into door
    assert state == (1, 2)
    assert reward == pytest.approx(8.0)
    assert env.has_key is True


def test_level_2_reach_goal_after_opening():
    """Test de la réussite au niveau 2 une fois le chemin libre."""
    maze = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    start = (1, 1)
    goal = (1, 2)
    env = MazeEnv(maze, start, goal, level=2)
    env.reset()
    env.has_key = True
    env.maze[goal] = 0
    state, reward, done, info = env.step(3)
    assert state == goal
    assert bool(done) is True
    assert reward == pytest.approx(REWARD_GOAL)


def test_render_level_2():
    """Test de la méthode render pour le niveau 2."""
    maze = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    env = MazeEnv(maze, start=(0, 0), goal=(2, 2), level=2)
    env.reset()
    try:
        env.render()
    except Exception as e:
        pytest.fail(f"La méthode render (niveau 2) a levé une exception: {e}")


def test_generate_maze_shape_and_values():
    """Test de la génération aléatoire du labyrinthe."""
    maze = generate_maze(width=9, height=7)
    assert maze.shape == (7, 9)
    assert maze[1, 1] == 0
    assert np.all(np.isin(maze, [0, 1]))
