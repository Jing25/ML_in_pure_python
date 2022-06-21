import numpy as np

class Linear:
    
    def forward(self, x, w, b):

        out = x @ w + b
        cache = (x, w, b)
        
        return out, cache
    
    def backward(self, grad, cache):
        x, w, b = cache
        
        dw = x.T @ grad
        db = grad.sum(axis=0)
        dx = grad @ w.T
        
        return dx, dw, db
    
    
class ReLU:

    def forward(self, x):
        out = np.clip(x, 0, None)
        cache = x
        return out, cache
    
    def backward(self, grad, cache):
        x = cache
        dx = (x > 0) * grad
        
        return dx
    
    
class LinearReLU:
    def __init__(self):
        self.linear = Linear()
        self.relu = ReLU()

    def forward(self, x, w, b):
        
        a, l1_cache = self.linear.forward(x, w, b)
        out, relu_cache = self.relu.forward(a)
        
        cache = (l1_cache, relu_cache)
        
        return out, cache
    
    def backward(self, grad, cache):
        
        l1_cache, relu_cache = cache
        
        dx = self.relu.backward(grad, relu_cache)
        dx, dw, db = self.linear.backward(dx, l1_cache)
        
        return dx, dw, db

def softmax_loss(y, y_pred):
    
    n_sample = len(y)
    
    shifted = y_pred - y_pred.max(axis=1, keepdims=True)
    Z = np.exp(shifted).sum(axis=1, keepdims=True)
    
    log_prob = shifted - np.log(Z)
    prob = np.exp(log_prob)
    loss = -log_prob[np.arange(n_sample), y].sum() / n_sample
    
    dx = prob.copy()
    dx[np.arange(n_sample), y] -= 1
    
    dx /= n_sample
    
    return loss, dx
    
    
    
class TwoLayerNet:
    
    def __init__(self, in_dim, hidden_dim, n_classes, lr, reg):
        
        self.lr = lr
        self.reg = reg
        self.params = {}
        
        self.params['w1'] = np.random.randn(in_dim, hidden_dim) / np.sqrt(2. / in_dim)
        self.params['w2'] = np.random.randn(hidden_dim, n_classes) / np.sqrt(2. / hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(n_classes)
        
    def loss(self, X, y=None):
        
        score, grads = 0, {}
        
        l1 = LinearReLU()
        l2 = Linear()
        
        l1_out, l1_cache = l1.forward(X, self.params['w1'], self.params['b1'])
        score, l2_cache = l2.forward(l1_out, self.params['w2'], self.params['b2'])
        
        if y is None:
            return score
        
        loss, dx = softmax_loss(y, score)
        loss += self.reg * ((self.params['w2'] * self.params['w2']).sum() + (self.params['w1'] * self.params['w1']).sum())
        
        dx, dw2, db2 = l2.backward(dx, l2_cache)
        dx, dw1, db1 = l1.backward(dx, l1_cache)
        
        grads['w1'] = dw1 + 2 * self.reg * self.params['w1']
        grads['w2'] = dw2 + 2 * self.reg * self.params['w2']
        grads['b1'] = db1
        grads['b2'] = db2
        
        return loss, grads