from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde, norm

@dataclass
class ExpConfig(object):
    beta: int =  0.5
    n_points: int =  300
    t_max: int =  5
    n_frames: int = 100
    x_max: float = 2.
    random_seed: int = 45
    initial_distribution: str =  'uniform'

@dataclass
class LiouvilleTheorem:
    """Класс исследования теоремы Лиувиля."""
    cfg = ExpConfig()
    fig = None
    animation = None

    def _get_initial_points(self):
        np.random.seed(self.cfg.random_seed)
        dist_type = self.cfg.initial_distribution

        if dist_type == 'gaussian':
            return np.random.normal(0, 1, self.cfg.n_points)

        elif dist_type == 'uniform':
            return np.random.uniform(-1, 1, self.cfg.n_points)

        else:
            raise ValueError("Неизвестный тип распределения")

    def _get_initial_density(self, x):
        dist_type = self.cfg.initial_distribution

        if dist_type == 'gaussian':
            return norm.pdf(x)
        elif dist_type == 'uniform':
            return np.where((x >= -1) & (x <= 1), 0.5, 0.0)
        else:
            raise ValueError("Неизвестный тип распределения")

    def animate_points_and_kde(self, save_path=None):
        """Анимация движения точек и их распределения KDE"""
        self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        beta = self.cfg.beta

        # Инициализация точек
        x0 = self._get_initial_points()

        # Настройка графиков
        self.fig.suptitle(f'Эволюция точек и KDE распределения (β = {beta})', fontsize=14)

        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-0.1, 0.1)
        ax1.set_yticks([])
        ax1.set_xlabel('x')
        ax1.set_title('Расположение точек')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlim(-3, 3)
        ax2.set_ylim(0, 3)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Плотность')
        ax2.set_title('KDE распределения')
        ax2.grid(True, alpha=0.3)

        # Графические элементы
        scat = ax1.scatter([], [], alpha=0.7, color='blue', s=50)
        line, = ax2.plot([], [], 'r-', linewidth=2)
        time_text = ax1.text(0.02, 0.92, '', transform=ax1.transAxes, fontsize=12,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        def init():
            scat.set_offsets(np.zeros((self.cfg.n_points, 2)))
            line.set_data([], [])
            time_text.set_text('')
            return scat, line, time_text

        def animate(frame):
            t = frame * self.cfg.t_max / self.cfg.n_frames
            x_current = np.exp(-t * beta) * x0

            # Обновление точек
            scat.set_offsets(np.column_stack([x_current, np.zeros(self.cfg.n_points)]))

            # Обновление KDE
            if len(x_current) > 1:
                kde = gaussian_kde(x_current)
                x_kde = np.linspace(-3, 3, 100)
                y_kde = kde(x_kde)
                line.set_data(x_kde, y_kde)

            time_text.set_text(f't = {t:.2f}')
            return scat, line, time_text

        self.animation = FuncAnimation(
            self.fig, animate, frames=self.cfg.n_frames,
            init_func=init, blit=True, interval=100
        )

        plt.tight_layout()

        if save_path:
            self.animation.save(save_path, writer='pillow', fps=20)

        plt.show()

    def animate_density(self, save_path=None):
        """Анимация эволюции плотности аналитического решения"""
        self.fig, ax = plt.subplots(figsize=(10, 5))
        beta = self.cfg.beta

        ax.set_xlim(-self.cfg.x_max, self.cfg.x_max)
        ax.set_ylim(0, 9)
        ax.set_xlabel('x')
        ax.set_ylabel('ρ(x,t)')
        ax.grid(True)

        line, = ax.plot([], [], lw=2, color='coral')

        def ro(x, t):
            return np.exp(beta * t) * self._get_initial_density(x * np.exp(beta * t))

        def init():
            line.set_data([], [])
            return line,

        def animate(t):
            x = np.linspace(-self.cfg.x_max, self.cfg.x_max, 1000)
            y = ro(x, t)
            line.set_data(x, y)

            # Динамическое обновление границ
            ax.set_ylim(0, max(5, np.max(y) * 1.1))
            ax.set_title(f'Эволюция плотности (t = {t:.2f}, β = {beta})')
            return line,

        self.animation = FuncAnimation(
            self.fig, animate,
            frames=np.linspace(0, self.cfg.t_max, self.cfg.n_frames),
            init_func=init, blit=True, interval=100, repeat=True
        )

        plt.tight_layout()

        if save_path:
            self.animation.save(save_path, writer='ffmpeg', fps=20)

        plt.show()


if __name__ == "__main__":
    lt = LiouvilleTheorem()

    lt.animate_points_and_kde(save_path='real_evolution.gif')
    lt.animate_density(save_path='density_evolution.gif')