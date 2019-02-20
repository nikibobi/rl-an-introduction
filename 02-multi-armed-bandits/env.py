import numpy as np
from abc import ABC, abstractmethod

class MultiArmedBandit(ABC):
    def __init__(self, n):
        self.n = n
        self.means = np.zeros(n)
    
    @abstractmethod
    def reset(self):
        pass

    def optimal(self):
        return np.argmax(self.means)

    def step(self, a):
        return np.random.randn() + self.means[a]
    
    def update(self):
        pass

class StationaryMAB(MultiArmedBandit):
    def reset(self):
        self.means = np.random.randn(self.n)

class NonStationaryMAB(MultiArmedBandit):
    def reset(self):
        mean = np.random.randn()
        self.means = np.full(self.n, mean)
    
    def update(self):
        self.means += np.random.normal(scale=0.01, size=self.means.shape)

def create_bandits(n, stationary=True):
    if stationary:
        return StationaryMAB(n)
    else:
        return NonStationaryMAB(n)