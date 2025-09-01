import numpy as np
import networkx as nx
import timeit


def mean_runtime(n: int = 10, func: callable = lambda: None, *args, **kwargs) -> float:
    times = []
    ress = []
    for _ in range(n):
        start = timeit.default_timer()
        res = func(*args, **kwargs)
        end = timeit.default_timer()
        times.append(end - start)
        ress.append(res)
    return np.mean(times), np.mean(ress)



class Solver:
    def __init__(self, n_drones: int, k: int, size: int, cell_size: float = 100):
        self.n_drones = n_drones
        self.k = k
        self.size = size

        self.cell_size = cell_size

        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)


    def to_graph(self):
        G = nx.Graph()

        directions = [
            (0, 1),   # right
            (1, 0),   # down
            (0, -1),  # left
            (-1, 0),  # up
            (1, 1),   # down-right
            (1, -1),  # down-left
            (-1, 1),  # up-right
            (-1, -1)  # up-left
        ]

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] > 0:
                    G.add_node((i, j))
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size and self.grid[ni, nj] > 0:
                            G.add_edge((i, j), (ni, nj))

        return G


    def random_pos(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
        # Randomly place drones in the grid
        positions = np.random.choice(self.size * self.size, self.n_drones)

        for pos in positions:
            i, j = divmod(pos, self.size)
            self.grid[i, j] += 1

    def get_connectivity_value(self) -> int:
        return nx.node_connectivity(self.to_graph())
    

    def mean_position(self):
        # Get indices of all ones
        rows, cols = np.where(self.grid > 0)
        
        # Compute mean row and mean column
        mean_row = rows.mean()
        mean_col = cols.mean()
        
        return round(mean_row), round(mean_col)
    
    def furthest_point(self, center_point: tuple[int, int]):
        # Get positions of ones
        rows, cols = np.where(self.grid > 0)

        # Compute distances from center point
        distances = np.sqrt((rows - center_point[0])**2 + (cols - center_point[1])**2)

        # Index of furthest point
        idx = np.argmax(distances)

        return int(rows[idx]), int(cols[idx])
    
    def get_v(self, center_point: tuple[int, int], point: tuple[int, int]) -> float:
        neighbors = []
        directions = [
            (0, 1),   # right
            (1, 0),   # down
            (0, -1),  # left
            (-1, 0),  # up
            (1, 1),   # down-right
            (1, -1),  # down-left
            (-1, 1),  # up-right
            (-1, -1)  # up-left
        ]
        for di, dj in directions:
            ni, nj = point[0] + di, point[1] + dj
            if 0 <= ni < self.size and 0 <= nj < self.size and self.grid[ni, nj] == 0:
                neighbors.append((ni, nj))
        
        min_dist = float('inf')
        best_neighbor = None

        for neighbor in neighbors:
            dist = np.linalg.norm(np.array(neighbor).astype(np.float32) - np.array(center_point).astype(np.float32))
            if dist < min_dist:
                min_dist = dist
                best_neighbor = neighbor

        return best_neighbor

    
    def solve(self):
        self.random_pos()


        p = 0
        t = 0


        p_max = self.size * self.size

        while self.get_connectivity_value() < self.k and p < p_max:
            center_point = self.mean_position()

            u = self.furthest_point(center_point)

            v = self.get_v(center_point, u)

            self.grid[u] -= 1
            self.grid[v] += 1

            duv = np.linalg.norm(np.array(u).astype(np.float32) - np.array(v).astype(np.float32)) * self.cell_size


            duv = np.round(duv)

            t += duv
            p += 1

        return t


def main():

    for size in range(50, 100, 10):
        solver = Solver(n_drones=20, k=3, size=size)

        solver.solve()


        mean_time, mean_res = mean_runtime(20, solver.solve)

        print(mean_time, mean_res)



if __name__ == "__main__":
    main()
