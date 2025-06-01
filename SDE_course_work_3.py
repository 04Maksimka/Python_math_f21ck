import numpy as np
import matplotlib.pyplot as plt

# Параметры модели
omega0 = 1.0  # Частота осциллятора
alpha = 0.6  # Коэффициент затухания
beta = 0.6  # Коэффициент нелинейности
sigma = 0.6  # Интенсивность шума
eps = 0.05  # Малый параметр

# Параметры симуляции
T = 1000.0  # Общее время
dt = 0.01  # Шаг времени
n_steps = int(T / dt)  # Число шагов
t = np.linspace(0, T, n_steps)  # Временная сетка

# Начальные условия
x0 = 1.0
v0 = 0.0

# Инициализация массивов
x = np.zeros(n_steps)
v = np.zeros(n_steps)
x[0] = x0
v[0] = v0

# Генерация случайных приращений Винера
dW = np.sqrt(dt) * np.random.randn(n_steps)

# Численное интегрирование (метод Эйлера-Маруямы)
for i in range(1, n_steps):
    dx = v[i - 1] * dt
    dv = (
            (-omega0 ** 2 * x[i - 1]
            + eps * (-alpha * v[i - 1] + beta * x[i - 1] ** 3)) * dt
            + eps * sigma * dW[i]
    )

    x[i] = x[i - 1] + dx
    v[i] = v[i - 1] + dv

# График координаты
plt.figure()
plt.plot(t, x, 'b', linewidth=0.5)
plt.title('Решение x(t)')
plt.xlabel('Время')
plt.ylabel('x(t)')
plt.grid(True)

# Фазовый портрет
plt.figure()
plt.plot(x, v, 'g', linewidth=0.5)
plt.title('Фазовый портрет')
plt.xlabel('x')
plt.ylabel('v')
plt.grid(True)

# Распределение координаты
plt.figure()
plt.hist(x, bins=50, density=True, color='orange', alpha=0.7)
plt.title('Распределение x')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.grid(True)

plt.show()