import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def brownian_motion(T: float, N: int, mu: float, sigma: float, seed: int = None):
    """
    Generate a Brownian motion process using the Euler-Maruyama method.
        T (float): Time horizon (in years)
        N (int): Number of time steps
        mu (float): Drift rate
        sigma (float): Volatility
        seed (int): Seed for the random number generator (default: None)
    """
    dt = T/N
    np.random.seed(seed)
    W = np.cumsum(np.random.normal(loc=0, scale=np.sqrt(dt), size=(1, N))[0])
    W = np.insert(W, 0, 0)
    return mu*T + sigma*W

def update(num):
    ax.clear()
    ax.plot(bm[:num])

# Example usage:
T = 1.0 # Time horizon (in years)
N = 252 # Number of time steps (assuming 252 trading days per year)
mu = 0.05 # Drift rate
sigma = 0.2 # Volatility

bm = brownian_motion(T, N, mu, sigma)
fig, ax = plt.subplots()


ani = FuncAnimation(fig, update, frames=range(1, len(bm)))
plt.show()