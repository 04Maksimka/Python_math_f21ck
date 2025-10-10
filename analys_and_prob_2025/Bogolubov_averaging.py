"""Скрипт исследования решения уравнения Дуффинга."""
import numpy as np
import matplotlib.pyplot as plt


def solve_equation(epsilon, t0, t_max, q0, v0, h=0.01):
    """
    Численно решает уравнение d^2q/dt^2 + q = epsilon * q^3 методом RK4.

    Параметры:
        epsilon (float): параметр нелинейности.
        t0 (float): начальное время.
        t_max (float): конечное время.
        q0 (float): начальное значение q.
        v0 (float): начальное значение dq/dt.
        h (float): шаг интегрирования.

    Возвращает:
        t (np.ndarray): массив времени.
        q (np.ndarray): массив значений q.
        v (np.ndarray): массив значений dq/dt.
    """
    n_steps = int((t_max - t0) / h) + 1
    t = np.linspace(t0, t_max, n_steps)
    q = np.zeros(n_steps)
    v = np.zeros(n_steps)
    q[0] = q0
    v[0] = v0

    for i in range(n_steps - 1):
        k1_q = h * v[i]
        k1_v = h * (-q[i] + epsilon * q[i] ** 3)

        k2_q = h * (v[i] + 0.5 * k1_v)
        k2_v = h * (-(q[i] + 0.5 * k1_q) + epsilon * (q[i] + 0.5 * k1_q) ** 3)

        k3_q = h * (v[i] + 0.5 * k2_v)
        k3_v = h * (-(q[i] + 0.5 * k2_q) + epsilon * (q[i] + 0.5 * k2_q) ** 3)

        k4_q = h * (v[i] + k3_v)
        k4_v = h * (-(q[i] + k3_q) + epsilon * (q[i] + k3_q) ** 3)

        q[i + 1] = q[i] + (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6
        v[i + 1] = v[i] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return t, q, v


def averaging(t, q0, epsilon):
    """Первое приближенное решение: q(t) = q0 cos((1 - epsilon * 3 * q0^2 / 8)t)"""
    omega = 1 - epsilon * 3 * q0 ** 2 / 8
    return q0 * np.cos(omega * t)


def approximation(t, q0, epsilon):
    """Второе приближенное решение: q(t) = q0 cos(t) + epsilon q0^3(3/8 t sin(t) - 1/32 (cos 3t - cos t))"""
    term1 = q0 * np.cos(t)
    term2 = epsilon * q0 ** 3 * (3 / 8 * t * np.sin(t) - 1 / 32 * (np.cos(3 * t) - np.cos(t)))
    return term1 + term2


# Параметры
epsilon = 0.1
t0 = 0
t_max = 20
q0 = 1.0
v0 = 0.0
h = 0.01

# Численное решение
t, q_numerical, v_numerical = solve_equation(epsilon, t0, t_max, q0, v0, h)

# Приближенные решения
q_approx1 = averaging(t, q0, epsilon)
q_approx2 = approximation(t, q0, epsilon)

# Визуализация
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, q_numerical, label='Численное решение', linewidth=2)
plt.plot(t, q_approx1, '--', label='Усреднение БК')
plt.plot(t, q_approx2, '--', label='Приближение : малый параметр')
plt.xlabel('Время')
plt.ylabel('q(t)')
plt.legend()
plt.title('Сравнение численного решения с приближенными')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, np.abs(q_numerical - q_approx1), label='Ошибка усреднения Бк')
plt.plot(t, np.abs(q_numerical - q_approx2), label='Ошибка приближения')
plt.xlabel('Время')
plt.ylabel('Абсолютная ошибка')
plt.legend()
plt.title('Ошибки приближенных решений')
plt.yscale('log')  # Логарифмическая шкала для лучшего отображения ошибок
plt.grid(True)

plt.tight_layout()
plt.show()
