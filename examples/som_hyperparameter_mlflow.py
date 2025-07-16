import numpy as np
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.sklearn_wrapper.som_wrapper import SOMWrapper

# A proper metrics for SOM need to be implemented like Quantization error.
# For now made a prototype of dummy scorer and evaluate method need to be coded with proper SOM metrics
def scorer(estimator, X):
    metrics = estimator.evaluate(X)
    return -metrics["quantization_error"]

if __name__ == "__main__":
    
    X = np.random.random((100, 3))
    
    ## Section 1: Without pipeline train and load
    # Train SOM
    som = SOMWrapper(width=10, height=10, input_dim=3, alpha=0.1, iterations=100)
    som.fit(X)
    som.save_model("/mnt/data/som_model")
    som.save_image("/mnt/data/som_output.png")
    
    # Load model and predict
    som_loaded = SOMWrapper.load_model("/mnt/data/som_model")
    predictions = som_loaded.predict(X)    
    
    
    ## Section 2: MLflow-Integrated Hyperparameter Tuning with Scikit-Learn Pipeline

    pipeline = Pipeline([
        ("som", SOMWrapper())
    ])

    param_grid = {
        "som__width": [10, 20],
        "som__height": [10, 20],
        "som__alpha": [0.1, 0.5],
        "som__iterations": [100, 200],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=[(slice(None), slice(None))],
        verbose=2
    )

    with mlflow.start_run(run_name="SOM_GridSearch"):
        grid_search.fit(X)

        best_estimator = grid_search.best_estimator_
        best_som = best_estimator.named_steps["som"]

        metrics = best_som.evaluate(X)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)

        best_som.save_image("best_som.png")
        best_som.save_weights("best_weights.npy")

        mlflow.log_artifact("best_som.png")
        mlflow.log_artifact("best_weights.npy")

        print("Best Params:", grid_search.best_params_)
        print("Metrics:", metrics)
        
        

