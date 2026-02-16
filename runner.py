from scripts import evaluate, process_data, train
from setup import launch_mlflow
import mlflow


if __name__ == '__main__':
    launch_mlflow()
    with mlflow.start_run(run_name="wow_experiment"):
        process_data.process_data()
        train.train()
        evaluate.evaluate()
