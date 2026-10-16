import os
import numpy as np


class Predictor:
    def __init__(self, backend='keras', seed=42, epochs=30):
        self.backend, self.seed, self.epochs = backend, seed, epochs

    def fit(self, x, y):
        # The scaler and all model parameters see training observations only.
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        z = np.clip((x-self.mean)/self.scale, -10, 10)
        if self.backend == 'ridge':
            z = np.column_stack([np.ones(len(z)), z])
            penalty = np.eye(z.shape[1]) * 10
            penalty[0, 0] = 0
            self.weights = np.linalg.solve(z.T@z + penalty, z.T@y)
        else:
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            import tensorflow as tf
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(self.seed)
            tf.config.experimental.enable_op_determinism()
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(x.shape[1],)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1),
            ])
            self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
            # Fixed epochs: no peeking at the test fold for early stopping.
            for _ in range(self.epochs):
                for start in range(0, len(z), 64):
                    self.model.train_on_batch(z[start:start+64], y[start:start+64]*100)
        return self

    def predict(self, x):
        z = np.clip((x-self.mean)/self.scale, -10, 10)
        if self.backend == 'ridge':
            return np.column_stack([np.ones(len(z)), z]) @ self.weights
        return self.model(z, training=False).numpy().ravel() / 100

    def save(self, directory):
        np.savez(directory / 'scaler.npz', mean=self.mean, scale=self.scale)
        if self.backend == 'keras':
            self.model.save(directory / 'model.keras')
        else:
            np.save(directory / 'weights.npy', self.weights)
