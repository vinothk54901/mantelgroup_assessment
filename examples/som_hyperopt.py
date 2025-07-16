import numpy as np
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pathlib import Path
from src.som.model import SelfOrganizingMap  
from src.som.config import SOMConfig        

input_data = np.random.random((100, 3))          
SAVE_IMAGE_PATH = "artifacts/som_map.png"

def load_data(path):
    return np.load(path)

def define_search_space():
    return {
        'alpha': hp.uniform('alpha', 0.01, 0.5),
        'width': hp.quniform('width', 5, 30, 1),
        'height': hp.quniform('height', 5, 30, 1),
        'iterations': hp.quniform('iterations', 100, 1000, 10)
    }

def objective(params):
    data = input_data
    config = SOMConfig(
        width=int(params['width']),
        height=int(params['height']),
        input_dim=data.shape[1],
        alpha=float(params['alpha']),
        iterations=int(params['iterations'])
    )

    som = SelfOrganizingMap(config)
    som.fit(data)
    qe = som.quantization_error(data)  

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("quantization_error", qe)

        Path(SAVE_IMAGE_PATH).parent.mkdir(parents=True, exist_ok=True)
        som.save_image(SAVE_IMAGE_PATH)
        mlflow.log_artifact(SAVE_IMAGE_PATH)

    return {'loss': qe, 'status': STATUS_OK}

def run_hyperopt():
    trials = Trials()
    best = fmin(
        fn=objective,
        space=define_search_space(),
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )
    print("Best SOM hyperparameters:", best)

if __name__ == "__main__":
    mlflow.set_experiment("SOM Quantization Benchmark")
    run_hyperopt()
