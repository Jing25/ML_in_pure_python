import numpy as np

class MeanSquaredErrorLoss:
    def __call__(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred

        return ((self.y - self.y_pred)**2).mean()
    
    def backward(self):
        n = np.prod(self.y.shape)
        return -2 * (self.y - self.y_pred) / n


class BinaryCrossEntropy:
    def __call__(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred

        return -(self.y * np.log(self.y_pred) + (1 - self.y) * np.log(1 - self.y_pred)).mean()

    def backward(self):
        n = np.prod(self.y.shape)
        return -(self.y * (1 / self.y_pred) + (self.y - 1) * (1 / (1 - self.y_pred))) / n
    
    
class CrossEntropy:
    def __call__(self, y, y_pred):
        """
        y: class number eg. [0, 1, 2] for 3 classes
        y_pred: size (n_sample, n_class), softmax scores
        """
        n_sample = y_pred.shape[0]
        self.y_pred = y_pred
        self.y = y
        
        return -(np.log(y_pred[np.arange(n_sample), y])).sum() / n_sample
    
    def backward(self):
        n_sample = len(self.y_pred)

        return -(1 / self.y_pred) / n_sample
    
    
class SoftmaxCrossEntropy:
    
    def __call__(self, y, y_pred):
        """
        y_pred: probability before softmax
        """
        n_sample = len(y)
        self.y = y
        shifted = y_pred - y_pred.max(axis=1, keepdims=True)
        z = np.exp(shifted).sum(axis=1, keepdims=True)
        log_prob = shifted - np.log(z)
        self.prob = np.exp(log_prob)
        
        return -log_prob[np.arange(n_sample), y].sum() / n_sample

    
    def backward(self):
        
        n_sample = len(self.y)
        dx = self.prob.copy()
        dx[np.arange(n_sample), self.y] -= 1
        
        return dx / n_sample
        
    def softmax(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        x -= x.max(axis=1, keepdims=True)
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    
class SVMLoss:
    
    def __call__(self, y, y_pred):
        
        n = len(y)
        self.y = y
        
        y_true = y_pred[np.arange(n), y].reshape(-1, 1)
        
        self.margins = np.clip(y_pred - y_true + 1, 0, None)
        self.margins[np.arange(n), y] = 0
        
        loss = self.margins.sum() / n
        
        return loss
    
    def backward(self):
        
        n = len(self.y)
        num_pos = (self.margins > 0).sum(axis=1)
        grad = np.zeros_like(self.margins)
        grad[self.margins > 0] = 1
        grad[np.arange(n), self.y] -= num_pos
        
        return grad / n
    
    
    
class BinaryClassificationMetrices:
    def confusion_matrics(self, y_true, y_pred):
        self.TP = (y_pred == y_true)[y_true == 1].sum()
        self.TN = (y_pred == y_true)[y_true == 0].sum()
        self.FP = len(y_pred[y_pred == 1]) - self.TP
        self.FN = len(y_pred[y_pred == 0]) - self.TN
        
        return [[self.TP, self.FP], [self.FN, self.TN]]
    
    def precision(self):
        "How accuracy the positive predictions are"
        return self.TP / (self.TP + self.FP)
    
    def recall(self):
        "The coverage of positive samples (sensitivity)"
        return self.TP / (self.TP + self.FN)
    
    def specificity(self):
        "The coverage of negative samples"
        return self.TN / (self.TN + self.FP)
    
    def F1(self):
        """
        Harmonic mean of precision and recall:
        n / (1/x1 + 1/x2 + ... + 1/xn)
        """
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)
        
        
    
        
        
        
        
