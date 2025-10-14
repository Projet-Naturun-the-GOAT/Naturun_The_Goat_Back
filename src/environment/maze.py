import numpy as np
import random


class MazeEnv:
    def __init__(self, maze, start, goal):
        """
        maze: 2D numpy array, 0=free, 1=wall
        start: (row, col) tuple
        goal: (row, col) tuple
        """
        self.maze = maze
        self.start = start
        self.goal = goal
        self.state = start
        self.n_steps = 0

    def reset(self):
        self.state = self.start
        self.n_steps = 0
        return self.state

    def step(self, action):
        """
        action: 0=up, 1=down, 2=left, 3=right
        returns: next_state, reward, done, info
        """
        old_state = self.state
        row, col = self.state
        if action == 0:  # up
            next_state = (row - 1, col)
        elif action == 1:  # down
            next_state = (row + 1, col)
        elif action == 2:  # left
            next_state = (row, col - 1)
        elif action == 3:  # right
            next_state = (row, col + 1)
        else:
            next_state = self.state

        # Check boundaries and walls
        if (
            0 <= next_state[0] < self.maze.shape[0]
            and 0 <= next_state[1] < self.maze.shape[1]
            and self.maze[next_state] == 0
        ):
            self.state = next_state
        else:
            next_state = self.state  # invalid move

        done = next_state == self.goal
        if done:
            reward = 20.0
        elif next_state == old_state:
            reward = -0.5  # penalty for hitting wall
        else:
            reward = -0.2  # penalty for other valid moves

        info = {}
        self.n_steps += 1

        return next_state, reward, done, info

    def render(self):
        maze_render = np.array(self.maze, dtype=str)
        maze_render[maze_render == "0"] = "."
        maze_render[maze_render == "1"] = "#"
        r, c = self.state
        gr, gc = self.goal
        if (r, c) == (gr, gc):
            maze_render[r, c] = "*"
        else:
            maze_render[r, c] = "A"
            maze_render[gr, gc] = "G"
        print("\n".join(" ".join(row) for row in maze_render))


def generate_maze(width=31, height=31):
    """Génère un labyrinthe aléatoire avec l'algorithme de backtracking récursif"""
    maze = np.ones((height, width), dtype=int)

    def carve(r, c):
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and maze[nr, nc] == 1:
                maze[nr - dr // 2, nc - dc // 2] = 0
                maze[nr, nc] = 0
                carve(nr, nc)

    maze[1, 1] = 0
    carve(1, 1)
    return maze


# Example usage:
if __name__ == "__main__":
    maze = generate_maze(11, 11)
    env = MazeEnv(maze, start=(0, 1), goal=(4, 4))
    env.render()
    env.step(1)  # down
    print("\nAfter one step down:")
    env.render()
    print(env.step(1))  # down
    print("\nAfter another step down:")
    env.render()
    print(env.step(1))  # down
    print("\nAfter another step down:")
    env.render()
