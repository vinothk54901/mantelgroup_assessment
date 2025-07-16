from pydantic import BaseModel, PositiveInt, confloat

class SOMConfig(BaseModel):
    
    """
    Configuration schema for initializing a Self-Organizing Map (SOM).

    Attributes:
        width (PositiveInt): grid width.
        height (PositiveInt): grid height.
        input_dim (PositiveInt): Dimensionality of input vectors.
        alpha (float): Initial learning rate, must be between (0, 1], default is 0.1.
        iterations (PositiveInt): Number of training iterations, default is 10.
    """

    width: PositiveInt
    height: PositiveInt
    input_dim: PositiveInt
    alpha: confloat(gt=0, le=1) = 0.1
    iterations: PositiveInt = 10
