import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------- 1. Исправленный Neural ODE слой -------------------
class NeuralODELayer(layers.Layer):
    def __init__(self, units, n_steps=20, T=1.0, dynamics_hidden=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.n_steps = n_steps
        self.T = T
        self.dt = T / n_steps
        self.dynamics_hidden = dynamics_hidden

    def build(self, input_shape):
        self.dynamics = keras.Sequential([
            layers.Dense(self.dynamics_hidden, activation='tanh'),
            layers.Dense(self.units)
        ])
        # Строим подсеть, передавая фиктивный вход нужной формы
        self.dynamics.build((None, self.units))
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, units) – начальное состояние
        batch_size = tf.shape(inputs)[0]
        dt = self.dt
        dynamics = self.dynamics

        # TensorArray для хранения всех состояний на траектории
        trajectory_ta = tf.TensorArray(
            dtype=inputs.dtype, size=self.n_steps,
            dynamic_size=False, clear_after_read=False
        )

        # Первоначальное состояние добавляем явно
        z = inputs
        for i in range(self.n_steps):
            trajectory_ta = trajectory_ta.write(i, z)
            dz = dynamics(z)
            z = z + dt * dz   # шаг Эйлера

        # Собираем все состояния в тензор (n_steps, batch, units)
        traj = trajectory_ta.stack()   # (n_steps, batch, units)
        # Переупорядочиваем в (batch, n_steps, units)
        traj = tf.transpose(traj, [1, 0, 2])
        return traj

# ------------------- 2. Данные -------------------
num_samples = 1000
input_dim = 10
num_classes = 3

X = np.random.randn(num_samples, input_dim).astype(np.float32)
y = np.random.randint(0, num_classes, size=(num_samples,))
y_onehot = keras.utils.to_categorical(y, num_classes)

split = int(0.8 * num_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_onehot[:split], y_onehot[split:]

# ------------------- 3. Модель -------------------
model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    NeuralODELayer(units=64, n_steps=30, T=2.0),
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# ------------------- 4. Обучение -------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# ------------------- 5. Тест -------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')