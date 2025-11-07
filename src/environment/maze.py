import numpy as np
import random


class MazeEnv:
    def __init__(self, maze, start, goal, level=1):
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
        self.level = level
        self.last_state = None

        if self.level == 2:
            self.has_key = False
            self.key_position = (1, maze.shape[1] - 2)
            self.maze[self.key_position] = 3
            self.door_positions = []
            gr, gc = goal
            # Place doors around the goal
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and maze[nr, nc] == 0:
                    self.maze[nr, nc] = 2
                    self.door_positions.append((nr, nc))
            self.visited = set()

    def reset(self):
        self.state = self.start
        self.n_steps = 0
        self.last_state = None
        if self.level == 2:
            self.has_key = False
            self.maze[self.key_position] = 3
            for pos in self.door_positions:
                self.maze[pos] = 2
            self.visited = set()
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

        reward = -.2  # default penalty
        done = False

        # Check boundaries and walls
        if (
            0 <= next_state[0] < self.maze.shape[0]
            and 0 <= next_state[1] < self.maze.shape[1]
            and self.maze[next_state] in [0, 2, 3]
        ):
            self.state = next_state
        else:
            next_state = self.state  # invalid move
        
        cell_value = self.maze[next_state]

        # Pénalité pour aller-retour immédiat
        if self.last_state is not None and next_state == self.last_state:
            reward -= 1.0

        if self.level == 2:
            if cell_value == 3 and not self.has_key:
                self.has_key = True
                self.maze[next_state] = 0
                self.state = next_state
                reward = 15.0
            elif cell_value == 2:
                if self.has_key:
                    self.maze[next_state] = 0
                    self.state = next_state
                    reward = 8.0
                else:
                    next_state = old_state
                    reward = -5.0
            elif cell_value == 0:
                self.state = next_state
                # Bonus exploration si nouvelle case
                if not hasattr(self, "visited"):
                    self.visited = set()
                if next_state not in self.visited:
                    reward += 0.2
                    self.visited.add(next_state)
            else:
                next_state = old_state
                reward = -.3
            done = self.state == self.goal and self.maze[self.goal] == 0
            if done:
                reward = 20.0
        else:
            # Niveau 1 : logique classique
            if cell_value == 0:
                self.state = next_state
            else:
                next_state = old_state
                reward = -.3
            done = self.state == self.goal
            if done:
                reward = 20.0

        info = {}
        self.n_steps += 1
        self.last_state = old_state

        return next_state, reward, done, info

    def render(self):
        if self.level == 1:
            self.render_level_1()
        elif self.level == 2:
            self.render_level_2()
        else:
            print("Niveau de rendu inconnu.")

    def render_level_1(self):
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

    def render_level_2(self):
        maze_render = np.array(self.maze, dtype=str)
        maze_render[maze_render == "0"] = "."  # path
        maze_render[maze_render == "1"] = "#"  # wall
        maze_render[maze_render == "2"] = "D"  # door
        maze_render[maze_render == "3"] = "🔑"  # key
        r, c = self.state  # agent position row, col
        gr, gc = self.goal  # goal position row, col
        # Set doors around the goal
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = gr + dr, gc + dc  # neighbor row, col
            # Check bounds and if it's not a wall
            if 0 <= nr < self.maze.shape[0] and 0 <= nc < self.maze.shape[1]:
                if self.maze[nr, nc] == 0:
                    maze_render[nr, nc] = "O"

        kr, kc = 1, self.maze.shape[1] - 2
        if (r, c) == (kr, kc):
            maze_render[r, c] = "A"

        if (r, c) == (gr, gc):
            maze_render[r, c] = "*"
        elif (r, c) == (kr, kc):
            maze_render[r, c] = "A"
        elif (r, c) == (dr, dc):
            maze_render[r, c] = "A"
        else:
            maze_render[r, c] = "A"
            maze_render[gr, gc] = "G"
        
        print("\n".join(" ".join(row) for row in maze_render))


def generate_maze(width=31, height=31):
    """Génère un labyrinthe aléatoire avec l'algorithme de backtracking récursif"""
    maze = np.ones((height, width), dtype=int)

    """Carve est une fonction récursive pour creuser des chemins dans le labyrinthe"""
    def carve(r, c):
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and maze[nr, nc] == 1:
                maze[nr - dr // 2, nc - dc // 2] = 0
                maze[nr, nc] = 0
                carve(nr, nc)

    maze[1, 1] = 0  # Start point
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
