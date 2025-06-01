import numpy as np
import matplotlib.pyplot as plt


class StochasticChain:
    def __init__(self, n=3, epsilon=0.1, temp1=1.0, temp_n=1.0, dt=0.01, t_final=100):
        """
        Инициализация цепи частиц со стохастической динамикой

        Параметры:
        n - количество частиц
        epsilon - параметр нелинейности
        T1, Tn - температуры для первой и последней частиц
        dt - шаг интегрирования
        t_final - общее время моделирования
        """
        self.n = n
        self.epsilon = epsilon
        self.T1 = temp1
        self.Tn = temp_n
        self.dt = dt
        self.t_final = t_final
        self.num_steps = int(t_final / dt)
        self.t = np.linspace(0, t_final, self.num_steps + 1)
        self.x = np.zeros((n, self.num_steps + 1))
        self.v = np.zeros((n, self.num_steps + 1))
        self.x0 = np.random.uniform(-0.5, 0.5, n)
        self.v0 = np.random.uniform(-0.1, 0.1, n)

    def set_initial_conditions(self, x0=None, v0=None):
        """Установка начальных условий"""
        if x0 is not None:
            self.x0 = np.array(x0)
        if v0 is not None:
            self.v0 = np.array(v0)

    def solve(self):
        """Реализация метода Эйлера-Маруямы для решения СДУ"""
        # Установка начальных условий
        self.x[:, 0] = self.x0
        self.v[:, 0] = self.v0

        # Интегрирование
        for i in range(self.num_steps):
            dW = np.random.normal(0, np.sqrt(self.dt), self.n)

            # Обновление скоростей
            # Первая частица
            dx1 = self.x[1, i] - self.x[0, i]
            self.v[0, i + 1] = self.v[0, i] + (
                        -self.x[0, i] + self.x[1, i] - self.v[0, i] - self.epsilon * dx1 ** 3) * self.dt + np.sqrt(
                2 * self.T1) * dW[0]

            # Последняя частица
            dxn = self.x[self.n - 1, i] - self.x[self.n - 2, i]
            self.v[self.n - 1, i + 1] = self.v[self.n - 1, i] + (
                        -self.x[self.n - 1, i] + self.x[self.n - 2, i] - self.v[
                    self.n - 1, i] - self.epsilon * dxn ** 3) * self.dt + np.sqrt(2 * self.Tn) * dW[self.n - 1]

            # Внутренние частицы
            for j in range(1, self.n - 1):
                dx_forward = self.x[j, i] - self.x[j + 1, i]  # x_j - x_{j+1}
                dx_backward = self.x[j - 1, i] - self.x[j, i]  # x_{j-1} - x_j
                self.v[j, i + 1] = self.v[j, i] + (
                        -2 * self.x[j, i] + self.x[j - 1, i] + self.x[j + 1, i]
                        - self.epsilon * dx_forward ** 3
                        + self.epsilon * dx_backward ** 3
                ) * self.dt + np.sqrt(self.epsilon) * dW[j]

            # Обновление координат
            self.x[:, i + 1] = self.x[:, i] + self.v[:, i] * self.dt

    def plot_trajectories(self):
        """Визуализация траекторий координат и скоростей"""
        plt.figure(figsize=(12, 8))

        plt.subplot(211)
        for i in range(self.n):
            plt.plot(self.t, self.x[i], label=f'$x_{i + 1}$')
        plt.xlabel('Время')
        plt.ylabel('Координата')
        plt.legend()
        plt.grid(True)
        plt.title('Траектории координат частиц')

        plt.subplot(212)
        for i in range(self.n):
            plt.plot(self.t, self.v[i], label=f'$v_{i + 1}$')
        plt.xlabel('Время')
        plt.ylabel('Скорость')
        plt.legend()
        plt.grid(True)
        plt.title('Траектории скоростей частиц')

        plt.tight_layout()

    def plot_phase_portraits(self):
        """Фазовые портреты для каждой частицы"""
        plt.figure(figsize=(15, 4 * self.n))
        for i in range(self.n):
            plt.subplot(self.n, 1, i + 1)
            plt.plot(self.x[i], self.v[i])
            plt.xlabel(f'$x_{i + 1}$')
            plt.ylabel(f'$v_{i + 1}$')
            plt.title(f'Фазовый портрет частицы {i + 1}')
            plt.grid(True)
        plt.tight_layout()

    def plot_energy(self, energy_type='kinetic'):
        """
        Визуализация энергий частиц

        Параметры:
        energy_type - 'kinetic' (v_i^2/2), 'potential' (x_i^2/2), или 'both'
        """
        plt.figure(figsize=(12, 8))

        if energy_type in ['kinetic', 'both']:
            for i in range(self.n):
                kinetic_energy = 0.5 * self.v[i] ** 2
                plt.plot(self.t, kinetic_energy, label=f'$K_{i + 1} = v_{i + 1}^2/2$')

        if energy_type in ['potential', 'both']:
            for i in range(self.n):
                potential_energy = 0.5 * self.x[i] ** 2
                plt.plot(self.t, potential_energy, '--', label=f'$U_{i + 1} = x_{i + 1}^2/2$')

        plt.xlabel('Время')
        plt.ylabel('Энергия')
        plt.legend()
        plt.grid(True)
        title = {
            'kinetic': 'Кинетическая энергия частиц',
            'potential': 'Потенциальная энергия частиц',
            'both': 'Энергии частиц'
        }[energy_type]
        plt.title(title)

    def plot_kinetic_energy_per_particle(self):
        """Визуализация кинетической энергии (v_i^2/2) для каждой частицы."""
        for i in range(self.n):
            plt.figure()
            kinetic_energy = self.v[i]**2
            plt.plot(self.t, kinetic_energy, 'r')
            mean_energy = np.mean(kinetic_energy)
            plt.axhline(y=mean_energy, color='b', linestyle='--',
                        label=f'Среднее = {mean_energy:.4f}')
            plt.xlabel('Время')
            plt.ylabel(f'$v_{i+1}^2/2$')
            plt.title(f'Кинетическая энергия частицы {i+1}')
            plt.legend()
            plt.grid(True)


if __name__ == '__main__':
    # Создание и настройка системы
    chain = StochasticChain(n=5, epsilon=0, temp1=0, temp_n=10, dt=0.01, t_final=1000)
    # Решение системы
    chain.solve()
    # Визуализация результатов
    chain.plot_kinetic_energy_per_particle()
    plt.show()
