import pygame
import numpy as np
import colorsys

def sqrt_ceil_nearest_square(n: int) -> int:
    root = int(np.ceil(np.sqrt(n)))
    return root

def get_pos_from_grid(grid: np.ndarray) -> np.ndarray:
    pos = np.argwhere(grid > 0)
    pos = pos[pos[:, 2].argsort()]

    return pos[:, :2]


class GridView:
    def __init__(self, size: int = 640, offset: int = 50, render_mode: str = "human", render_fps: int = 60):
        pygame.init()
        pygame.display.set_caption("Drone Environment")
        self.size = size
        self.offset = offset

        self.render_mode = render_mode
        self.render_fps = render_fps
        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode((self.size, self.size)) if render_mode == "human" else None

    def to_pos(self, row: int, col: int, cell_size: int, center = True):
        x = col * cell_size + self.offset
        y = row * cell_size + self.offset
        if center:
            x += cell_size // 2
            y += cell_size // 2

        return (x, y)

    def calc_edges(self, nodes: np.ndarray):
        edges = []
        for i in range(len(nodes)-1):
            i_row, i_col = nodes[i]
            for j in range(i+1, len(nodes)):
                j_row, j_col = nodes[j]

                if abs(float(i_row) - float(j_row)) <= 1 and abs(float(i_col) - float(j_col)) <= 1:
                    edges.append((i, j))

        return edges

    def get_cell_size(self, n: int):
        cell_size = (self.size - 2 * self.offset) // n
        return cell_size

    def draw_grid(self, surf: pygame.Surface, grid: np.ndarray):
        n = grid.shape[0]

        cell_size = self.get_cell_size(n)

        for i in range(n+1):
            # vertical lines
            last_pos = self.offset + n * cell_size
            pygame.draw.line(surf, (200, 200, 200), (self.offset, self.offset + i * cell_size), (last_pos, self.offset + i * cell_size), 2)

            # horizontal lines
            pygame.draw.line(surf, (200, 200, 200), (self.offset + i * cell_size, self.offset), (self.offset + i * cell_size, last_pos), 2)

    def draw_nodes(self, surf: pygame.Surface, grid: np.ndarray):

        n = grid.shape[0]
        cell_size = self.get_cell_size(n)

        nodes = get_pos_from_grid(grid)

        n_drones = len(nodes)

        edges = self.calc_edges(nodes)

        node_colors = np.array([colorsys.hsv_to_rgb(i/n_drones, 1, 1) for i in range(n_drones)]) * 255

        drone_cells = np.argwhere(grid.sum(axis=2) > 0)


        node_pos = np.zeros((n_drones, 3))

        for row, col in drone_cells:
            drone_idxs = np.where(grid[row, col])[0]

            colors = node_colors[drone_idxs]

            minicell_n = sqrt_ceil_nearest_square(len(colors))

            minicell_size = cell_size // minicell_n

            cell_x, cell_y = self.to_pos(row, col, cell_size=cell_size, center=False)


            for i in range(len(colors)):
                idx = minicell_n * minicell_n - i - 1
                cell_row = idx // minicell_n
                cell_col = idx % minicell_n

                x = cell_x + cell_col * minicell_size + minicell_size // 2
                y = cell_y + cell_row * minicell_size + minicell_size // 2

                node_pos[drone_idxs[i]] = [x, y, minicell_size // 3]

        # this is for drawing edges
        for i, j in edges:
            x1, y1 = node_pos[i][:2]
            x2, y2 = node_pos[j][:2]

            line_width = min(node_pos[i][2], node_pos[j][2]) * 0.7
            if line_width < 1:
                line_width = 1

            pygame.draw.line(surf, (0, 0, 0, 50), (x1, y1), (x2, y2), int(line_width))

        # this is for drawing nodes
        for i in range(n_drones):
            x, y, r = node_pos[i]
            pygame.draw.circle(surf, node_colors[i], (int(x), int(y)), int(r))



    def render(self, grid: np.ndarray[tuple[int, int, int], np.bool_]):
        surf = pygame.Surface((self.size, self.size))
        surf.fill((255, 255, 255))

        self.draw_grid(surf=surf, grid=grid)

        self.draw_nodes(surf, grid)

        surf = pygame.transform.flip(surf, False, True)

        return self.post_render(surf)

    def post_render(self, surf: pygame.Surface):
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            self.screen.fill((255, 255, 255))
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return self._create_image_array(surf, (self.size, self.size))
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )


def main():
    # np.random.seed(0)
    grid_size = 4
    n = 5

    # pos = np.random.randint(0, grid_size, size=(n, 2))
    # grid = np.zeros((grid_size, grid_size, n), dtype=np.bool_)
    # grid[pos[:, 0], pos[:, 1], np.arange(n)] = True

    # print(grid.sum(axis=2))

    # print(np.argwhere(grid.sum(axis=2) > 0))

    # print(np.where(grid[3, 3]))

    view = GridView(size=640, render_mode="human", render_fps=1)

    for i in range(10):
        grid_size = np.random.randint(4, 10)

        pos = np.random.randint(0, grid_size, size=(n, 2))


        grid = np.zeros((grid_size, grid_size, n), dtype=np.bool_)

        grid[pos[:, 0], pos[:, 1], np.arange(n)] = True

        print(pos)
        view.render(grid=grid)




if __name__ == "__main__":
    main()
