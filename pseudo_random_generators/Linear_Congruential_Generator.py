from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class LCG:
    seed: int = 42
    m: int = field(default=2 ** 31 - 1, init=False)  # modulus
    a: int = field(default=1103515245, init=False)  # multiplier
    c: int = field(default=12345, init=False)  # increment
    state: int = field(init=False)

    def __post_init__(self):
        self.state = self.seed % self.m  # initial state

    def next(self) -> float:
        # Generate the next pseudorandom number
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m  # Normalize to [0, 1)

    def generate(self, n: int) -> list:
        # Generate a list of n pseudorandom numbers
        return [self.next() for _ in range(n)]


# Example usage
if __name__ == "__main__":
    lcg_x = LCG()
    lgc_y = LCG(seed=1234)
    lgc_z = LCG(seed=125)
    x = lcg_x.generate(1000)
    y = lcg_x.generate(1000)
    plt.scatter(x, y)
    plt.show()
