import matplotlib.pyplot as plt
from tensorflow import keras


def display_learning_curves(
    history: keras.callbacks.History, save_name: str = "training_graphs"
) -> None:
    """Plot the Loss and Accuracy for a Training Run."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")

    plt.savefig(f"{save_name}.png")
