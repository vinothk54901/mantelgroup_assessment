from sklearn.base import BaseEstimator
import numpy as np
from config import SOMConfig
from som import SelfOrganizingMap
from pydantic import ValidationError
import joblib
from typing import Optional, Union


class SOMWrapper(BaseEstimator):
    """
    A Scikit-learn compatible wrapper for Self-Organizing Map (SOM) that supports
    integration with pipelines and hyperparameter tuning.

    Attributes:
        width (int): Width of the SOM grid.
        height (int): Height of the SOM grid.
        input_dim (int): Dimensionality of input data.
        alpha (float): Initial learning rate.
        iterations (int): Number of training iterations.
        som (SelfOrganizingMap): Internal SOM instance.
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        input_dim: int = 3,
        alpha: float = 0.1,
        iterations: int = 100,
    ):
        """
        Initializes the SOM wrapper with configuration parameters.
        """
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.alpha = alpha
        self.iterations = iterations
        self.som = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fits the SOM model to the input data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, input_dim).
            y (Optional[np.ndarray]): Ignored; included for sklearn compatibility.

        Returns:
            self: Trained instance of SOMWrapper.
        """
        try:
            config = SOMConfig(
                width=self.width,
                height=self.height,
                input_dim=self.input_dim,
                alpha=self.alpha,
                iterations=self.iterations,
            )
        except ValidationError as e:
            print(f"[ERROR] Invalid SOM config:\n{e}")
            raise

        self.som = SelfOrganizingMap(config)
        self.som.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Return BMU positions for each input vector
        return np.array([self.som._find_bmu(vec) for vec in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict logic
        return self.transform(X)

    def evaluate(self, X: np.ndarray):
        # Place holder for evaluation logic with quantization
        # or topographic error
        return {}

    def save_weights(self, path="weights.npy"):
        np.save(path, self.som.weights)

    def load_weights(self, path="weights.npy"):
        # load weights logic here
        weights = np.load(path)
        self.som.weights = weights

    def save_model(self, path: str = "som_model"):
        joblib.dump(self, path)

    @classmethod
    def load_model(cls, path: str = "som_model"):
        return joblib.load(path)

    def save_image(self, path="som_output.png"):
        self.som.save_image(path)
