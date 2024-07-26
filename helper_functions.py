import numpy as np
from tensorflow import keras


def compile_and_train_model(
    model: keras.models.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
) -> keras.callbacks.History:
    """Compile and Train a Model, Return the History Callback Object."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    return model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )
