import numpy as np
from som.model import SelfOrganizingMap
from som.config import SOMConfig

if __name__ == '__main__':
    input_data = np.random.random((100, 3))
    config = SOMConfig(width=20, height=20, input_dim=3, alpha=0.1, iterations=100)
    som = SelfOrganizingMap(config)
    som.fit(input_data)
    som.save_image("./output.png")
    
    

