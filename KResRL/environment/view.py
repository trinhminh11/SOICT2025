import pygame
import numpy as np
import colorsys

class GridView:
    def __init__(self, n_row: int, n_col: int, cell_size: int = 50, offset: int = None, render_mode: str = "human", render_fps: int = 60):
        pygame.init()
        pygame.display.set_caption("Drone Environment")
        self.n_row = n_row
        self.n_col = n_col
        self.cell_size = cell_size
        self.off_set = offset if offset is not None else self.cell_size

        self.width = (self.n_col) * self.cell_size + 2 * self.off_set
        self.height = (self.n_row) * self.cell_size + 2 * self.off_set

        self.render_mode = render_mode
        self.render_fps = render_fps
        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode((self.width, self.height)) if render_mode == "human" else None
    
    def to_pos(self, row, col):
        x = col * self.cell_size + self.off_set + self.cell_size // 2
        y = row * self.cell_size + self.off_set + self.cell_size // 2
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



    def draw_grid(self, surf: pygame.Surface):
        for i in range(self.n_row+1):
            pygame.draw.line(surf, (200, 200, 200), (self.off_set, self.off_set + i * self.cell_size), (self.width - self.off_set, self.off_set + i * self.cell_size), 2)

        for j in range(self.n_col+1):
            pygame.draw.line(surf, (200, 200, 200), (self.off_set + j * self.cell_size, self.off_set), (self.off_set + j * self.cell_size, self.height - self.off_set), 2)
        
    def draw_nodes(self, surf: pygame.Surface, nodes: np.ndarray):
        for i, (row, col) in enumerate(nodes):
            x, y = self.to_pos(row, col)

            color = colorsys.hsv_to_rgb(i/len(nodes), 1, 1)
            pygame.draw.circle(surf, ([color[0] * 255, color[1] * 255, color[2] * 255]), (x, y), radius=self.cell_size // 3)

    def draw_edges(self, surf: pygame.Surface, nodes: np.ndarray):
        edges = self.calc_edges(nodes)
        for i, j in edges:
            x1, y1 = self.to_pos(*nodes[i])
            x2, y2 = self.to_pos(*nodes[j])
            
            pygame.draw.line(surf, (0, 0, 0), (x1, y1), (x2, y2), 10)

    def render(self, nodes: np.ndarray):
        
        surf = pygame.Surface((self.width, self.height))
        surf.fill((255, 255, 255))

        self.draw_grid(surf=surf)

        self.draw_edges(surf, nodes)
        self.draw_nodes(surf, nodes)

        surf = pygame.transform.flip(surf, False, True)

        return self.post_render(surf)

    def post_render(self, surf: pygame.Surface):
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            self.screen.fill(255)
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            
        elif self.render_mode == "rgb_array":
            return self._create_image_array(surf, (self.width, self.height))
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")
        
    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )


def main():
    view = GridView(n_row=10, n_col=10, cell_size=50, render_mode="human", render_fps=1)


    for i in range(10):
        view.render(nodes=np.random.randint(0, 10, size=(20, 2)))

if __name__ == "__main__":
    main()
