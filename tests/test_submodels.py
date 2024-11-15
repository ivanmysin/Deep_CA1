import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate

# Параметры данных
n_samples = 1000
timesteps = 20
input_dim = 10

# Генерация случайных входных данных
X = np.random.rand(n_samples, timesteps, input_dim)

# Генерация меток (например, бинарная классификация)
y = np.random.randint(0, 2, size=(n_samples,))

def create_submodel(input_layer):
    x = GRU(units=64, return_sequences=True)(input_layer)
    x = Dense(32, activation='relu')(x)
    return x

# Определяем общий входной слой
inputs = Input(shape=(timesteps, input_dim))

# Создаем три параллельные ветви
branch_1 = create_submodel(inputs)
branch_2 = create_submodel(inputs)
branch_3 = create_submodel(inputs)

# Конкатенируем выходные данные всех ветвей
merged = Concatenate()([branch_1, branch_2, branch_3])

# Добавляем финальные слои
x = GRU(units=128, return_sequences=False)(merged)
outputs = Dense(1, activation='sigmoid')(x)

# Создаем полную модель
model = Model(inputs=inputs, outputs=outputs)

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
