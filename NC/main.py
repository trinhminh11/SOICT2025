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

        # Find the point with value > 0 that is closest to the mean position
        min_dist = float('inf')
        closest_point = None
        for r, c in zip(rows, cols):
            dist = np.sqrt((r - mean_row)**2 + (c - mean_col)**2)
            if dist < min_dist:
                min_dist = dist
                closest_point = (r, c)
        return closest_point
    
    def furthest_point(self, center_point: tuple[int, int]):
        # Get positions of ones
        rows, cols = np.where(self.grid > 0)

        # Compute distances from center point
        distances = np.sqrt((rows - center_point[0])**2 + (cols - center_point[1])**2)

        # Indices of points with maximum distance
        max_dist = np.max(distances)
        indices = np.where(distances == max_dist)[0]
        furthest_points = [(int(rows[i]), int(cols[i])) for i in indices]
        return furthest_points
    
    def get_u_v(self, center_point: tuple[int, int], points: list[tuple[int, int]]):
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

        best_point = None
        best_neighbor = None
        min_dist = float('inf')

        for point in points:
            neighbors: list[tuple[int, int]] = []
            for di, dj in directions:
                ni, nj = point[0] + di, point[1] + dj
                if 0 <= ni < self.size and 0 <= nj < self.size and self.grid[ni, nj] == 0:
                    neighbors.append((ni, nj))
            

            for neighbor in neighbors:
                dist = np.linalg.norm(np.array(neighbor).astype(np.float32) - np.array(center_point).astype(np.float32))
                if dist < min_dist:
                    min_dist = dist
                    best_neighbor = neighbor
                    best_point = point

        return best_point, best_neighbor

    
    def solve(self, pos = None):
        if pos is None:
            self.random_pos()
        else:
            self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
            for p in pos:
                self.grid[p[0], p[1]] += 1


        p = 0
        t = 0


        p_max = self.size * self.size

        while self.get_connectivity_value() < self.k and p < p_max:
            center_point = self.mean_position()

            u_s = self.furthest_point(center_point)

            u, v = self.get_u_v(center_point, u_s)


            self.grid[u] -= 1
            self.grid[v] += 1


            duv = np.linalg.norm(np.array(u).astype(np.float32) - np.array(v).astype(np.float32)) * self.cell_size

            duv = np.round(duv, 10)


            t += duv
            p += 1

        return t


def main():

    solver = Solver(n_drones=10, k=3, size=10, cell_size=1)

    pos = [[ 4,  8],
         [48, 46],
         [ 0,  3],
         [41,  8],
         [45, 10],
         [22, 40],
         [29, 23],
         [ 1,  5],
         [34, 23],
         [ 0, 25],
         [35, 17],
         [ 3, 22],
         [40, 11],
         [38, 18],
         [41, 16],
         [37, 49],
         [17,  4],
         [16,  9],
         [16, 23],
         [10, 46]]

    pos = [[5, 5],
       [9, 6],
       [3, 0],
       [7, 2],
       [3, 9],
       [2, 2],
       [9, 4],
       [7, 4],
       [2, 1],
       [6, 6]]
    
    res = solver.solve(pos)

    # mean_time, mean_res = mean_runtime(2, solver.solve, pos)
    # print(mean_time, mean_res)

    print(res)

    print(solver.grid)
    print(solver.get_connectivity_value())



if __name__ == "__main__":
    main()
