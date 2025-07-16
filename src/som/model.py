import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from .config import SOMConfig

class SelfOrganizingMap:
    def __init__(self, config: SOMConfig):
        self.config = config
        self.weights = self._init_weights()
        self._init_coordinate_grid()

    def _init_weights(self) -> np.ndarray:
        return np.random.random((self.config.width, self.config.height, self.config.input_dim))

    def _init_coordinate_grid(self):
        """
        Precompute the X and Y coordinates of the SOM grid to speed up BMU neighborhood calculations.
        """
        self.x_grid, self.y_grid = np.meshgrid(
            np.arange(self.config.width),
            np.arange(self.config.height),
            indexing="ij"
        )
    
    def _decay(self, initial: float, t: int) -> float:
        decay_constant = self.config.iterations / np.log(initial)
        return initial * np.exp(-t / decay_constant)

    def _find_bmu(self, vector: np.ndarray) -> Tuple[int, int]:
        distances = np.sum((self.weights - vector) ** 2, axis=2)
        bmu_x,bmu_y = np.unravel_index(np.argmin(distances), (self.config.width, self.config.height))
        return bmu_x,bmu_y
    
    
    ##To increase perfomance moving out of traditional forloop and adapting with numpy broad casting.
    # def _update_weights(self, vector: np.ndarray, bmu_x:int,bmu_y:int, alpha: float, sigma: float):
    #     for x in range(self.config.width):
    #         for y in range(self.config.height):
    #             dist = np.linalg.norm([x - bmu_x, y - bmu_y])
    #             influence = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    #             self.weights[x, y] += alpha * influence * (vector - self.weights[x, y])
                
    def _update_weights(self, vector: np.ndarray, bmu_x: int, bmu_y: int, alpha: float, sigma: float):
        """
        Vectorized weight update using Gaussian neighborhood influence.
        """
        # Distance squared from BMU to all neurons
        dist_sq = (self.x_grid - bmu_x) ** 2 + (self.y_grid - bmu_y) ** 2

        # Compute neighborhood influence as a Gaussian
        influence = np.exp(-dist_sq / (2 * sigma ** 2))[..., np.newaxis]

        # Broadcast the update across all neurons
        self.weights += alpha * influence * (vector - self.weights)            
                
    def _get_decays(self, t):
        alpha = self._decay(self.config.alpha, t)
        sigma = self._decay(max(self.config.width, self.config.height) / 2, t)
        return alpha, sigma

    def _process_vector(self, vector, alpha, sigma):
        bmu_x, bmu_y = self._find_bmu(vector)
        self._update_weights(vector, bmu_x, bmu_y, alpha, sigma)

    def fit(self, data: np.ndarray):
        for t in range(self.config.iterations):
            alpha, sigma = self._get_decays(t)
            for vector in data:
                self._process_vector(vector, alpha, sigma)
                
    def quantization_error(self, data: np.ndarray) -> float:
        """
        Computes the quantization error (QE) for the trained SOM.
        
        QE is defined as the average Euclidean distance between each input vector
        and its corresponding BMU's weight vector.

        Args:
            data (np.ndarray): Input data used to compute QE.

        Returns:
            float: Quantization error value.
        """
        total_error = 0.0
        for vector in data:
            bmu_x, bmu_y = self._find_bmu(vector)
            bmu_weight = self.weights[bmu_x, bmu_y]
            total_error += np.linalg.norm(vector - bmu_weight)
        return total_error / len(data)

    def save_image(self, path: str):
        plt.imsave(path, self.weights)
