import numpy as np

def decode_state(features_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    n_drones = len(features_state)

    adjacency_matrix = features_state[:, :n_drones]

    
    node_features = features_state[:, n_drones:]

    return node_features, adjacency_matrix




