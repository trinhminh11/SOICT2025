from typing import Literal
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from munkres import Munkres

from KResRL.environment.view import GridView


class KRes(gym.Env):
    ACTION_MAP = {
        0: (0, 0),      # Nothing
        1: (-1, 0),     # Up
        2: (-1, 1),     # Up right
        3: (0, 1),      # Right
        4: (1, 1),      # Down right
        5: (1, 0),      # Down
        6: (1, -1),     # Down left
        7: (0, -1),     # Left
        8: (-1, -1)     # Up left
    }

    NODE_FEATURES = [
        "first_center_row",     # normalize (0, 1)  first_center_row/n_cols
        "first_center_col",     # normalize (0, 1)  first_center_col/n_rows
        "row",                  # normalize (0, 1)  row/n_cols
        "col",                  # normalize (0, 1)  col/n_rows
        # "center_row",           # normalize (0, 1)  center_row/n_cols
        # "center_col",           # normalize (0, 1)  center_col/n_rows
        # "step_size",            #           (0, 1)  1 / size
        # "total_movement",       # normalize (0, 1)  total_movement/ size
        # "degree",               # normalize (0, 1)  degree/ K
        "node_connectivity",    # normalize (0, 1)  node_connectivity/K
    ]

    GLOBAL_FEATURES = [
        # "graph_connectivity", # normalize (0, 1)  graph_connectivity/K

    ]

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": 60,
    }

    def __init__(self, n_drones: int, k: int, size: int, return_adj: bool = True, return_state: Literal["grid", "pos", "features"] = "features", normalize_features: bool = True, alpha = 0.1, render_mode: Literal["human", "rgb_array"] = None, render_fps: int = 60) -> None:
        super().__init__()
        self.n_drones = n_drones
        self.k = k
        self.size = size
        self.max_movement = size  # max movement of 1 drone in grid steps
        
        self.return_adj = return_adj
        self.return_state = return_state
        self.normalize_features = normalize_features
        self.alpha = alpha

        self.view = None # view for rendering
        self.render_mode = render_mode
        self.render_fps = render_fps


        if self.return_state == "grid":
            self.observation_space = spaces.Box(low=0, high=2, shape = (size, size, self.n_drones), dtype=np.uint16) # n_drones in a grid
        elif self.return_state == "pos":
            self.observation_space = spaces.Box(low=0, high=size - 1, shape=(self.n_drones, 2), dtype=np.uint16)
        elif self.return_state == "features":
            if self.normalize_features:
                low = 0
                high = 1
            else:
                low = 0
                high = np.inf

            # adjacency matrix + node_features + global_features
            # adjacency matrix = space[:, :self.n_drones]
            # node_features = space[:, self.n_drones:self.n_drones + len(self.NODE_FEATURES)]
            # global_features = space[:, self.n_drones + len(self.NODE_FEATURES):]
            if return_adj:
                shape = (self.n_drones, self.n_drones + len(self.NODE_FEATURES))
            else:
                shape = (self.n_drones, len(self.NODE_FEATURES))
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                shape=shape,
            )
        else:
            raise ValueError("Invalid return_state value. Use 'grid' or 'pos'.")

        self.action_space = spaces.MultiDiscrete([9]*self.n_drones)  # 9 actions for each drone (stay, up, UR, right, RD, down, DL, left, LU)

        self.node_features = {
            "first_center_row": np.zeros((self.n_drones,), dtype=np.float32),
            "first_center_col": np.zeros((self.n_drones,), dtype=np.float32),
            "row": np.zeros((self.n_drones,), dtype=np.uint16),
            "col": np.zeros((self.n_drones,), dtype=np.uint16),
            "center_row": np.zeros((self.n_drones,), dtype=np.float32),
            "center_col": np.zeros((self.n_drones,), dtype=np.float32),
            "step_size": np.full((self.n_drones,), 1/(self.size-1), dtype=np.float32),
            "total_movement": np.zeros((self.n_drones,), dtype=np.uint16),
            "node_connectivity": np.zeros((self.n_drones,), dtype=np.uint16),
            "degree": np.zeros((self.n_drones,), dtype=np.uint16)
        }

        self.normalize_node: dict[str, float] = {
            "first_center_row": self.size - 1 if self.normalize_features else 1,
            "first_center_col": self.size - 1 if self.normalize_features else 1,    
            "row": self.size - 1 if self.normalize_features else 1,
            "col": self.size - 1 if self.normalize_features else 1,
            "center_row": self.size - 1 if self.normalize_features else 1,
            "center_col": self.size - 1 if self.normalize_features else 1,
            "step_size": 1,
            "total_movement": self.max_movement if self.normalize_features else 1,
            "node_connectivity": self.k if self.normalize_features else 1,
            "degree": self.k if self.normalize_features else 1
        }

        self.global_features = {
            "graph_connectivity": 0,
        }

        self.normalize_global: dict[str, float] = {
            "graph_connectivity": self.k if self.normalize_features else 1,
        }

        # all the initial_state can only be create a reset
        self.initial_drone_pos: np.ndarray[tuple[int, int], np.uint16] = None
        self.initial_grid_state: np.ndarray[tuple[int, int, int], np.uint16] = None
        self.initial_graph_state: nx.Graph = None
        self.initial_features_state: np.ndarray[tuple[int, int], np.float32] = None
        self.initial_global_features  = self.global_features.copy()

        # all of these attributes below should only be set in drone_pos.setter and initial_drone_pos.setter
        self._drone_pos: np.ndarray[tuple[int, int], np.uint16] = None
        self.grid_state: np.ndarray[tuple[int, int, int], np.uint16] = None
        self.graph_state: nx.Graph = None


    # handling different kind of state

    def __get_graph_state(self, drone_pos: np.ndarray[tuple[int, int], np.uint16]):
        G = nx.Graph()

        for i in range(self.n_drones):
            G.add_node(i)
            for j in range(i+1, self.n_drones):
                i_pos = drone_pos[i]
                j_pos = drone_pos[j]

                i_diff = i_pos[0] - j_pos[0] if i_pos[0] >= j_pos[0] else j_pos[0] - i_pos[0]
                j_diff = i_pos[1] - j_pos[1] if i_pos[1] >= j_pos[1] else j_pos[1] - i_pos[1]

                if i_diff <= 1 and j_diff <= 1:
                    G.add_edge(i, j)
        return G
    
    def __get_grid_state(self, drone_pos: np.ndarray[tuple[int, int], np.uint16]):
        _state = np.zeros((self.size, self.size, self.n_drones), dtype=np.uint16)
        for i in range(self.n_drones):
            _state[drone_pos[i][0], drone_pos[i][1], i] = 1
        return _state

    def __get_features_state(self, graph_state: nx.Graph, node_features: dict[str, np.ndarray]):
        if self.return_adj:
            shape = (self.n_drones, self.n_drones + len(self.NODE_FEATURES))
            s = self.n_drones
            adj = nx.adjacency_matrix(graph_state).todense().astype(np.float32)
        else:
            shape = (self.n_drones, len(self.NODE_FEATURES))
            s = 0
            adj = []

        _features = np.zeros(shape, dtype=np.float32)

        _features[:, :s] = adj

        _features[:, s:] = np.array([node_features[feature] / self.normalize_node[feature] for feature in self.NODE_FEATURES], dtype=np.float32).T

        return _features

    @property
    def features_state(self):
        return self.__get_features_state(self.graph_state, self.node_features)

    @property
    def drone_pos(self) -> np.ndarray[tuple[int, int], np.uint16]:
        return self._drone_pos
    
    @drone_pos.setter
    def drone_pos(self, value: np.ndarray[tuple[int, int], np.uint16]) -> None:
        cost_matrix = np.zeros((self.n_drones, self.n_drones), dtype=np.uint16)
        for i in range(self.n_drones):
            for j in range(self.n_drones):
                diff = np.linalg.norm(self.initial_drone_pos[i].astype(np.float32) - value[j].astype(np.float32))
                cost_matrix[i, j] = np.ceil(diff/np.sqrt(2))

        indexes = Munkres().compute(cost_matrix.copy())
        # Reassign drone_pos so that each drone moves to the position assigned by the optimal matching
        after_matching_drone_pos = np.zeros_like(value)
        for i, j in indexes:
            after_matching_drone_pos[i] = value[j]

        # self._drone_pos = after_matching_drone_pos
        self._drone_pos = value

        self.graph_state = self.__get_graph_state(self._drone_pos)
        self.grid_state = self.__get_grid_state(self._drone_pos)


        for i in range(self.n_drones):
            self.node_features["row"][i] = self._drone_pos[i][0]
            self.node_features["col"][i] = self._drone_pos[i][1]
            self.node_features["total_movement"][i] = cost_matrix[i][indexes[i][1]]
            self.node_features["node_connectivity"][i] = self.get_drone_connectivity_value(i)
            self.node_features["degree"][i] = self.graph_state.degree[i]

        center = self.get_grid_center()
        self.node_features["center_row"] = np.full((self.n_drones,), center[0], dtype=np.float32)
        self.node_features["center_col"] = np.full((self.n_drones,), center[1], dtype=np.float32)

        self.global_features["graph_connectivity"] = min(self.node_features["node_connectivity"])


    # only called at reset
    def __set_initial(self, drone_pos):
        self.initial_drone_pos = drone_pos
        self.initial_graph_state = self.__get_graph_state(drone_pos)
        self.initial_grid_state = self.__get_grid_state(drone_pos)

        _node_features = self.node_features.copy()

        for i in range(self.n_drones):
            _node_features["row"][i] = self.initial_drone_pos[i][0]
            _node_features["col"][i] = self.initial_drone_pos[i][1]
            _node_features["total_movement"][i] = 0
            _node_features["node_connectivity"][i] = self.get_drone_connectivity_value(i, self.initial_graph_state)
            _node_features["degree"][i] = self.initial_graph_state.degree[i]

        self.initial_global_features = self.global_features.copy()
        self.initial_global_features["graph_connectivity"] = min(_node_features["node_connectivity"])

        self.initial_features_state = self.__get_features_state(self.initial_graph_state, _node_features)

    @property
    def initial_state(self):
        if self.return_state == "grid":
            return self.initial_grid_state
        elif self.return_state == "pos":
            return self.initial_drone_pos
        elif self.return_state == "features":
            return self.initial_features_state

    @property
    def state(self):
        if self.return_state == "grid":
            return self.grid_state
        elif self.return_state == "pos":
            return self.drone_pos
        elif self.return_state == "features":
            return self.features_state

        raise ValueError("Invalid return_state value.")

    
    def __get_pretty_grid_state(self, drone_pos):
        _state = np.full((self.size, self.size), "", dtype="U255")
        for i in range(self.n_drones):
            x, y = drone_pos[i]
            _state[x, y] += " " + str(i) if _state[x, y] else str(i)
        return _state

    @property
    def pretty_initial_grid_state(self):
        return self.__get_pretty_grid_state(self.initial_drone_pos)

    @property
    def pretty_grid_state(self):
        return self.__get_pretty_grid_state(self.drone_pos)    


    def reset(self, seed=None, options=None):
        _drone_pos = np.zeros((self.n_drones, 2), dtype=np.uint16)  # Initialize drone positions
        # Randomly place drones in the grid
        positions = np.random.choice(self.size * self.size, self.n_drones)

        for drone, pos in enumerate(positions):
            i, j = divmod(pos, self.size)
            _drone_pos[drone] = (i, j)

        self.__set_initial(_drone_pos)

        self.drone_pos = _drone_pos

        self.last_center = self.get_grid_center()
        self.first_center = self.last_center

        # TODO: delete
        self.last_distance_to_center = np.zeros(self.n_drones, dtype=np.float32)
        self.last_distance_to_first_center = np.zeros(self.n_drones, dtype=np.float32)
        for i in range(self.n_drones):
            self.last_distance_to_center[i] = np.linalg.norm(self.drone_pos[i].astype(np.float32) - np.array(self.last_center).astype(np.float32))
            self.last_distance_to_first_center[i] = np.linalg.norm(self.drone_pos[i].astype(np.float32) - np.array(self.first_center).astype(np.float32))

        self.node_features["first_center_row"] = np.full((self.n_drones,), self.first_center[0], dtype=np.float32)
        self.node_features["first_center_col"] = np.full((self.n_drones,), self.first_center[1], dtype=np.float32)
        self.node_features["center_row"] = np.full((self.n_drones,), self.last_center[0], dtype=np.float32)
        self.node_features["center_col"] = np.full((self.n_drones,), self.last_center[1], dtype=np.float32)

        return self.state, {}
    
    def get_grid_center(self):
        grid_state = self.grid_state.sum(axis=2)
        rows, cols = np.where(grid_state > 0)
        mean_row = rows.mean()
        mean_col = cols.mean()
        return float(mean_row), float(mean_col)

    def step(self, action: np.ndarray):
        reward = 0
        info = {
            "initial_features": self.initial_features_state,
            "initial_global_features": self.initial_global_features,
            "K": self.k,
        }

        moves = np.array([self.ACTION_MAP[a] for a in action])

        # update node_features
        self.drone_pos = np.clip(self.drone_pos + moves, [0,0], [self.size-1, self.size-1])

        current_k = int(min(self.node_features["node_connectivity"]))
        info["current_k"] = current_k
        info["sparse_reward"] = np.zeros(self.n_drones, dtype=np.float32)
        info["global_features"] = self.global_features

        done = current_k >= self.k

        team_reward = 0

        # Compute team reward
        # team_reward = current_k - self.k//2 - self.alpha * sum(self.node_features["total_movement"]) + self.n_drones * 10 * done
        # center = self.get_grid_center()
        # info["center"] = center

        # # Compute individual drone rewards
        # drone_reward = np.zeros(self.n_drones, dtype=np.float32)
        # for i in range(self.n_drones):
        #     drone_reward[i] = self.node_features["node_connectivity"][i] / self.k + 10 * done

        #     distance_to_center = np.linalg.norm(self.drone_pos[i].astype(np.float32) - np.array(center).astype(np.float32))
            
        #     distance_penalty = -distance_to_center / self.size

        #     drone_reward[i] +=  distance_penalty
        #     info["sparse_reward"][i] = drone_reward[i]

        #     team_reward += distance_penalty

        # reward = team_reward + np.mean(drone_reward)




        center = self.get_grid_center()

        distance_to_last_center = np.zeros(self.n_drones, dtype=np.float32)
        distance_to_center = np.zeros(self.n_drones, dtype=np.float32)
        distance_to_first_center = np.zeros(self.n_drones, dtype=np.float32)


        for i in range(self.n_drones):
            distance_to_last_center[i] = np.linalg.norm(self.drone_pos[i].astype(np.float32) - np.array(self.last_center, dtype=np.float32))
            distance_to_center[i] = np.linalg.norm(self.drone_pos[i].astype(np.float32) - np.array(center, dtype=np.float32))
            distance_to_first_center[i] = np.linalg.norm(self.drone_pos[i].astype(np.float32) - np.array(self.first_center, dtype=np.float32))

        diff_distance = self.last_distance_to_center - distance_to_last_center
        diff_first_center_distance = self.last_distance_to_first_center - distance_to_first_center


        reward += diff_distance.sum() + 2*diff_first_center_distance.sum()

        self.last_distance_to_center = distance_to_center
        self.last_distance_to_first_center = distance_to_first_center
        self.last_center = center


        reward += 100*done + 4*current_k/self.k

        return self.state, reward, done, False, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)
    
    def _render(self, mode: str):
        assert mode in KRes.metadata["render_modes"]

        if self.view is None:
            self.view = GridView(
                n_row=self.size,
                n_col=self.size,
                cell_size=50,
                offset=50,
                render_mode=mode,
                render_fps=self.render_fps
            )
        
        
        return self.view.render(self.drone_pos)


    def close(self):
        pass

    def get_drone_connectivity_value(self, idx: int, G: nx.Graph = None):

        G = G if G is not None else self.graph_state
        nodes = list(G.nodes)
        n = len(nodes)

        if n <= 1:
            return 0.0
        if not nx.is_connected(G):
            return 0.0

        if n != self.n_drones:
            return 0.0

        min_connectivity = float("inf")
        
        for j in range(n):
            if j == idx:
                continue
            u, v = nodes[idx], nodes[j]
            k = nx.node_connectivity(G, s=u, t=v)
            min_connectivity = min(min_connectivity, k)

        return min_connectivity

    def matching(self):
        cost_matrix = np.zeros((self.n_drones, self.n_drones), dtype=np.uint16)

        for i in range(self.n_drones):
            for j in range(self.n_drones):

                diff = np.linalg.norm(self.initial_drone_pos[i].astype(np.float32) - self.drone_pos[j].astype(np.float32))

                cost_matrix[i, j] = np.ceil(diff/np.sqrt(2))

        indexes = Munkres().compute(cost_matrix.copy())

        return indexes, cost_matrix

if __name__ == "__main__":
    env = KRes(n_drones=5, k=2, size=4, normalize_features=False)
    obs, _ = env.reset()
    print("Initial observation:\n", obs)

    print(env.pretty_grid_state)

    sample_action = env.action_space.sample()
    print("Sample action:\n", sample_action)

    new_obs, reward, done, truncated, info = env.step(sample_action)

    print("New observation:\n", env.pretty_grid_state)

    for row, col in env.drone_pos:
        print(row, col)

    print(new_obs)

    env.close()