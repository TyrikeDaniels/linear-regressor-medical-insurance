from pathlib import Path

# import numpy as np
import pandas as pd
import typer
# from matplotlib import pyplot as plt

from lin_mod.config import PROCESSED_DATA_DIR
from lin_mod.modeling.linear_model import SGDRegressorCustom

app = typer.Typer()

@app.command()
def run(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    #model_path: Path = MODELS_DIR / "model.pkl",
):
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)

    linear_model = SGDRegressorCustom()

    linear_model.configure(
        epochs=1000,
        loss_function="mean_squared",
        learning_rate=1e-3, # 0.001
        regularization=(),
        logging=True
    )

    _, _, j_history = linear_model.fit(features.to_numpy(), labels.to_numpy().flatten(), 32)

#     plot(j_history)
#
#
# def plot(j_history) -> None:
#     """
#     Plot cost function evolution during optimization.
#
#     Args:
#         j_history (List[float]): List containing cost history
#     """
#     fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
#     ax1.plot(j_history)
#     ax2.plot(100 + np.arange(len(j_history[100:])), j_history[100:])
#
#     ax1.set_title("Cost vs. iteration")
#     ax1.set_xlabel("iteration step")
#     ax1.set_ylabel("Cost")
#
#     ax2.set_title("Cost vs. iteration (tail)")
#     ax2.set_xlabel("iteration step")
#     ax2.set_ylabel("Cost")
#
#     plt.show()
#

if __name__ == "__main__":
    app()