from typing import Callable
import numpy as np
from . import losses as mll

class ReLU:
    def __call__(self, x):
        self.x = x
        return np.clip(x, 0, None)
    
    def predict(self, x):
        return np.clip(x, 0, None)
    
    def backward(self, gradient):
        
        return (self.x > 0) * gradient

class Sigmoid:
    def __call__(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def predict(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, gradient):
        """
        ds = s(1 - s)
        """
        return gradient * self.y * (1 - self.y)
    
def softmax(x):
    """
    exp(x + logC) / sum(exp(x + logC))
    logC = max(x) along class dimention
    """
        
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    x -= x.max(axis=1, keepdims=True)  ## numeric stability
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

class SGD:
    def __init__(self, rho=0):
        self.v = 0
        self.rho = rho
        
    def update(self, dw):
        self.v = self.rho * self.v + dw
        return self.v
        
class PMSProp:
    def __init__(self, decay_rate=0.99, epsilon=1e-7):
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.grad_squared = 0
        
    def update(self, dw):
        self.grad_squared = self.decay_rate * self.grad_squared + (1 - self.decay_rate) * dw * dw
        return dw / (self.grad_squared**0.5 + self.epsilon)
    
class Adam:
    def __init__(self, beta1=0.9, beta2=0.99, epsilon=1e-8):
        self.moment1 = 0
        self.moment2 = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
    def update(self, dw):
        self.t += 1
        
        self.moment1 = self.beta1 * self.moment1 + (1 - self.beta1) * dw
        self.moment2 = self.beta2 * self.moment2 + (1 - self.beta2) * dw * dw
        
        moment1_unbias = self.moment1 / (1 - self.beta1 ** self.t)
        moment2_unbias = self.moment2 / (1 - self.beta2 ** self.t)
                
        return moment1_unbias / (moment2_unbias**0.5 + self.epsilon)
           
        
class Linear:
    def __init__(self, in_dim, out_dim=1, optim_rule: str='SGD', config: dict={'rho': 0.}):
        """
        in_dim: number of input features
            if x.shape = [1000, 10], in_dim = 10
        out_dim: number of output features 
            out_dim = y.shape[1]
        """
        self.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
        self.bias = np.zeros(out_dim)
        
        if optim_rule == 'SGD':
            rho = config['rho']
            self.optim = SGD(rho)
            
        elif optim_rule == 'RMSProp':
            decay_rate = config['decay_rate']
            epsilon = config['epsilon']
            self.optim = PMSProp(decay_rate, epsilon)
            
        elif optim_rule == 'Adam':
            beta1 = config['beta1']
            beta2 = config['beta2']
            epsilon = config['epsilon']
            self.optim = Adam(beta1, beta2, epsilon)

    
    def __call__(self, X):
        self.x = X
        return self.x @ self.weights + self.bias
    
    def predict(self, x):
        return x @ self.weights + self.bias

    def backward(self, gradient):
        self.dw = self.x.T @ gradient
        self.db = gradient.sum(axis=0)
        self.dx = gradient @ self.weights.T
        return self.dx

    def update(self, lr, reg=0):
        """
        Using L2 regularization
        """
        dw = self.optim.update(self.dw)
        self.weights -= lr * (dw + 2 * reg * self.weights) 
        self.bias -= lr * self.db

        
        
class LinearReLU:
    def __init__(self, in_dim, out_dim, optim_rule: str='SGD', config={'rho': 0.}):
        self.linear = Linear(in_dim, out_dim, optim_rule, config)
        self.relu = ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

    def predict(self, x):
        x = self.linear.predict(x)
        x = self.relu.predict(x)
        return x
    
    def backward(self, gradient):
        grad = self.relu.backward(gradient)
        grad = self.linear.backward(gradient)
        return grad
    
    def update(self, lr, reg):
        self.linear.update(lr, reg=reg)
        
class Dropout:
    
    def forward(self, x, p, mode='train'):
        n, m = x.shape
        self.mode = mode
        
        if mode == 'train':
            self.mask = (np.random.rand(n, m) < p) / p
            out = self.mask * x
            
        elif mode == 'test':
            out = x
            
        return out
    
    def backward(self, grad):
        if self.mode == 'train':
            grad = grad * self.mask
            
        elif self.mode == 'test':
            grad = grad
            
        return grad

        
class BatchNorm:
    def __init__(self, in_dim, momentum=0.1, eps=1e-5):
        self.running_mean = 0
        self.running_var = 0
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(in_dim)
        self.beta = np.zeros(in_dim)
        
    def __call__(self, x, gamma=None, beta=None, mode='train'):
        if gamma is not None and beta is not None:
            self.gamma = gamma
            self.beta = beta
            
        if mode == 'train':
            b_mean = x.mean(axis=0)
            b_var = x.var(axis=0)
            b_std = (b_var + self.eps) ** 0.5
            self.var = b_var
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * b_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * b_var

            self.x_norm = (x - b_mean) / b_std
            
            y = self.gamma * self.x_norm + self.beta
            
        else:
            x_norm = (x - self.running_mean) / (self.running_var + self.eps)**0.5
            y = self.gamma * x_norm + self.beta
        
        return y
    
    def backward(self, grad):
        n = len(self.x_norm)
        
        self.dgamma = (grad * self.x_norm).sum(axis=0)
        self.dbeta = grad.sum(axis=0)
        dx_norm = grad * self.gamma
        dx = (dx_norm - dx_norm.sum(0) / n - (dx_norm * self.x_norm).sum(0) * self.x_norm / n) / self.var**0.5
        
        return dx
        
    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta