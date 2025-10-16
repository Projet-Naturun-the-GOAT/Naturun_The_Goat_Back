import pytest
import numpy as np
from src.environment.maze import MazeEnv

REWARD_GOAL = 20.0
REWARD_WALL = -0.5
REWARD_MOVE = -0.2


@pytest.fixture
def setup_maze():
    """Fixture to initialize the maze environment."""
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
    """Test of the reset method."""
    env = setup_maze
    initial_state = env.reset()
    assert (
        initial_state == env.start
    ), "The initial state should correspond to the starting point."


def test_move_up(setup_maze):
    """Test of upward movement."""
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
    ), "The agent cannot go beyond the maze boundaries."
    assert (
        reward == REWARD_MOVE
    ), "Moving to a valid cell without reaching the goal should give a small penalty."


def test_move_down(setup_maze):
    """Test of downward movement."""
    env = setup_maze
    env.reset()
    state, reward, done, info = env.step(1)  # action = 1 (down)
    assert state == (1, 0), "The agent should move down one cell."
    assert (
        reward == REWARD_MOVE
    ), "Moving to a valid cell without reaching the goal should give a small penalty."


def test_move_left(setup_maze):
    """Test of leftward movement."""
    env = setup_maze
    env.start = (
        2,
        1,
    )  # Position the agent at (2,1) to test leftward movement
    env.reset()
    state, reward, done, info = env.step(2)  # action = 2 (left)
    assert state == (2, 0), "The agent should move to the left."
    assert (
        reward == REWARD_MOVE
    ), "Moving to a valid cell without reaching the goal should give a small penalty."


def test_move_right(setup_maze):
    """Test of rightward movement."""
    env = setup_maze
    env.start = (
        2,
        0,
    )  # Position the agent at (2,0) to test rightward movement
    env.reset()
    state, reward, done, info = env.step(3)  # action = 3 (right)
    assert state == (2, 1), "The agent should move to the right."
    assert (
        reward == REWARD_MOVE
    ), "Moving to a valid cell without reaching the goal should give a small penalty."


def test_hit_wall(setup_maze):
    """Test of movements into a wall."""
    env = setup_maze
    env.reset()
    state, reward, done, info = env.step(2)  # action = 2 (left) (movement into a wall)
    assert state == (0, 0), "The agent cannot move into a wall."
    assert (
        reward == REWARD_WALL
    ), "The agent should receive a penalty when hitting a wall."


def test_reach_goal(setup_maze):
    """Test when the agent reaches the goal."""
    maze = np.array([[0, 1], [0, 1]])
    start = (0, 0)
    goal = (1, 0)
    env = MazeEnv(maze, start, goal)
    env.reset()
    # Move to the goal (down, down, right, right)
    state, reward, done, info = env.step(1)  # right (reach the goal)

    assert done is True, "The agent should have reached the goal."
    assert state == env.goal, "The agent should reach the goal position."
    assert (
        reward == REWARD_GOAL
    ), "The agent should receive a large reward for reaching the goal."


def test_step_invalid_move(setup_maze):
    """Test to verify that the agent does not move when encountering a wall."""
    env = setup_maze
    env.reset()
    old_state = env.state
    # Test if the agent tries to move outside the maze boundaries
    state, reward, done, info = env.step(0)  # up (invalid movement into a wall)
    assert (
        state == old_state
    ), "The agent should not be able to move into a blocked area."
    assert (
        reward == REWARD_WALL
    ), "The agent should be penalized if it tries to move into a wall."


def test_render(setup_maze):
    """Test of the render method."""
    env = setup_maze
    env.reset()
    try:
        env.render()  # Just verify that it does not raise an exception
    except Exception as e:
        pytest.fail(f"The render method raised an exception: {e}")
