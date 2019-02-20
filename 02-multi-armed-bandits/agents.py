import numpy as np
from abc import abstractmethod, ABC

class Agent(ABC):
    def __init__(self, eps, q0=0):
        self.eps = eps
        self.q0 = q0
    
    def init(self, actions):
        self.actions = actions
        self.q = np.full(actions, self.q0, dtype=np.float)
    
    def act(self):
        if np.random.rand() > self.eps:
            return np.argmax(self.q)
        else:
            return np.random.choice(self.actions)
    
    @abstractmethod
    def step_size(self, a):
        pass

    def update(self, a, r):
        self.q[a] += self.step_size(a) * (r - self.q[a])

    @abstractmethod
    def legend(self):
        pass

class SampleAverageAgent(Agent):
    def __init__(self, eps, q0=0):
        super().__init__(eps, q0)
    
    def init(self, actions):
        super().init(actions)
        self.n = np.zeros(actions)

    def step_size(self, a):
        return 1 / self.n[a]
    
    def update(self, a, r):
        self.n[a] += 1
        super().update(a, r)
    
    def legend(self):
        return r'$\epsilon = {}, Q_0 = {}$ Sample Average'.format(self.eps, self.q0)

class ConstantStepAgent(Agent):
    def __init__(self, eps, alpha, q0=0):
        super().__init__(eps, q0)
        self.alpha = alpha

    def step_size(self, a):
        return self.alpha
    
    def legend(self):
        return r'$\epsilon = {}, \alpha = {}, Q_0 = {}$ Constant Step'.format(self.eps, self.alpha, self.q0)

class UCBAgent(SampleAverageAgent):
    def __init__(self, c):
        super().__init__(eps=0)
        self.c = c

    def act(self):
        return np.argmax(self.q + self.c * np.sqrt(np.log(np.sum(self.n)) / self.n))

    def legend(self):
        return r'$c = {}$ Upper-Confidence-Bound'.format(self.c)