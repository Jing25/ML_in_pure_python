from typing import Callable
import numpy as np
import collections
from . import losses as mll
from . import components as c

def euclidean_distance(arrA, arrB):  
    if arrA.ndim == 1 and arrB.ndim == 1:
        return ((arrA - arrB)**2).sum()
        
    if arrB.ndim == 1:
        return ((arrA - arrB)**2).sum(axis=1, keepdims=True)
    
    sq_arrA = (arrA * arrA).sum(axis=1, keepdims=True)
    sq_arrB = (arrB * arrB).sum(axis=1)
    
    arrA_arrB = arrA @ arrB.T
    
    return sq_arrA - 2 * arrA_arrB + sq_arrB

def manhattan_distance(arrA, arrB):
    """
    Only work when arrA and arrB are 2D matrices 
    """
    dist = np.abs(arrA[:, :, None] - arrB.T[None,:])
    return dist.sum(axis=1)
    
    

class KMeans:
    def __init__(self, k=3, seed=None):
        self.k = k
        self.seed = seed
    
    def init_centroids(self, data):
        if self.seed is not None:
            np.random.seed(self.seed)
        n = len(data)
        indices = np.random.choice(range(n), size=(self.k,))
        self.centroids = data[indices]
    
    def update_centroids(self, data):
        centroids = []
        for i in range(self.k):
            cluster = data[self.clusters == i]
            centroids.append(cluster.mean(axis=0))
            
        return np.array(centroids)
    
    def stop_criterion(self, centroids):
        if np.array_equal(centroids, self.centroids):
            return True
        self.centroids = centroids
    
    def assign_clusters(self, data, mode=None):
        if self.distance == 'euclidean':
            simi_matrix = euclidean_distance(data, self.centroids)
        
        if mode == 'predict':
            clusters = np.argmin(simi_matrix, axis=1) 
            return clusters
            
        self.clusters = np.argmin(simi_matrix, axis=1)    
        
    def predict(self, data):
        
        clusters = self.assign_clusters(data, mode='predict')
        return clusters
        
    
    def fit(self, data, distance: str='euclidean', max_iter=200):
        self.init_centroids(data)
        self.distance = distance
        
        niter, centroids = 0, None
        for niter in range(max_iter):
            self.assign_clusters(data)
            centroids = self.update_centroids(data)
            if self.stop_criterion(centroids):
                break
            
#             if niter % 10 == 0:
#                 print(f'interate {niter}')
        
        print(f'Done! Iterate {niter} steps')
        
    def score(self, data):
        scores = 0
        for i in range(self.k):
            c = data[self.clusters == i]
            if self.distance == 'euclidean':
                score = euclidean_distance(c, self.centroids[i])
            scores += score.sum()
        
        return -scores
                
            
        
class KNN:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def predict(self, x, k):
        from collections import Counter
        
        y_pred = []  ## size k, x size k
        similarity = euclidean_distance(x, self.x)
        indices = np.argpartition(similarity, k, axis=1)[:, :k]  ## k x n
        for i in range(len(x)):
            classes_count = Counter(self.y[indices[i]])
            pred = classes_count.most_common()[0][0]
            y_pred.append(pred)
            
        return y_pred
    
    @staticmethod
    def score(y, y_pred):
        accuracy = (y_pred == y).mean() * 100
        return accuracy
    
    def cross_validation(self, num_folds=5, k_choices=None):
        x_train_folds = np.array_split(self.x, num_folds, axis=0)
        y_train_folds = np.array_split(self.y, num_folds, axis=0)
        
        k_acc = {}
        for k in k_choices:
            accuracy = []
            for i in range(num_folds):
                x_train = np.concatenate(x_train_folds[:i] + x_train_folds[i+1:])
                y_train = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])

                x_val, y_val = x_train_folds[i], y_train_folds[i]

                knn = KNN(x_train, y_train)
                y_pred = knn.predict(x_val, k)
                acc = knn.score(y_val, y_pred)
                accuracy.append(acc)
                
            k_acc[k] = accuracy
            
        return k_acc            

### ------------------------ Decision Tree -------------------------------
class Leaf:
    def __init__(self, class_count):
        self.prediction = class_count
        
class DecisionNode:
    def __init__(self, partition, left, right):
        self.partition = partition
        self.left = left
        self.right = right
        
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))
        
class DecisionTree:
    
    @staticmethod
    def unique_value(data, col):
        return set([data[i][col] for i in range(len(data))])
    
    @staticmethod
    def is_numeric(value):
        return isinstance(value, (int, float))
    
    @staticmethod
    def count_label(data):
        count = collections.Counter()
        
        for i in range(len(data)):
            label = data[i][-1]
            count[label] += 1
            
        return count
        
    
    @classmethod
    def partition(cls, data, col, value):
        
        left = []
        right = []
        
        for i in range(len(data)):
            if cls.is_numeric(value):
                if data[i][col] <= value:
                    left.append(data[i])
                else:
                    right.append(data[i])
            else:
                if data[i][col] == value:
                    left.append(data[i])
                else:
                    right.append(data[i])
                    
        return left, right
    
    @classmethod
    def gini_impurity(cls, rows):
        class_count = cls.count_label(rows)
        total = len(rows)
        
        gini = 1
        for _, n in class_count.items():
            p = n / total
            gini -= p**2
            
        return gini
    
    @classmethod
    def info_gain(cls, left, right, impurity):
        gini_left = cls.gini_impurity(left)
        gini_right = cls.gini_impurity(right)
        
        p = len(left) / (len(left) + len(right))
        
        return impurity - (p * gini_left + (1 - p) * gini_right)

    
    @classmethod
    def best_partition(cls, data):
        
        most_gain = 0
        best_partition = None
        
        current_gini = cls.gini_impurity(data)
        
        for col in range(len(data[0]) - 1):
            features = cls.unique_value(data, col)
            
            for feature in features:
                
                left, right = cls.partition(data, col, feature)
                
                if len(left) == 0 or len(right) == 0:
                    continue
                                
                info_gain = cls.info_gain(left, right, current_gini)
                
                if most_gain <= info_gain:
                    most_gain = info_gain
                    best_partition = [col, feature]
                    
        return most_gain, best_partition
                
        
    def build_tree(self, data):
        
        gain, partition = self.best_partition(data)
        
        if gain == 0:
            class_count = self.count_label(data)
            return Leaf(class_count)
        
        left, right = self.partition(data, partition[0], partition[1])
        
        left_branch = self.build_tree(left)
        right_branch = self.build_tree(right)
        
        return DecisionNode(partition, left_branch, right_branch)
    
    def classify(self, row, node):
        if isinstance(node, Leaf):
            return node.prediction
        
        col, value = node.partition
        if self.is_numeric(value):
            if row[col] >= value:
                node = self.classify(row, node.left)
            else:
                node = self.classify(row, node.right)
        else:
            if row[col] == value:
                node = self.classify(row, node.left)
            else:
                node = self.classify(row, node.right)
                
        return node
        
    
    @classmethod
    def print_tree(cls, node, spacing=""):

        if isinstance(node, Leaf):
            print (spacing + "Predict", node.prediction)
            return

        # Print the question at this node
        print (spacing + str(node.partition))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        cls.print_tree(node.left, spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        cls.print_tree(node.right, spacing + "  ")
        
class LinearRegression:
    def __init__(self, in_dim, out_dim):
        
        self.linear = c.Linear(in_dim, out_dim)
        self.MSEloss = mll.MeanSquaredErrorLoss()     
        
    def forward(self, X):
        return self.linear(X)
    
    def predict(self, X):
        return self.linear.predict(X)
    
    def backward(self, gradient):
        self.linear.backward(gradient)

    def update(self, lr, reg):
        self.linear.update(lr, reg)
    
    def fit(self, X, y, lr, n_iter, reg=0, log=True, n=5, X_val=None, y_val=None):

        train_loss = []
        train_acc = []
        val_acc = []
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for i in range(n_iter):
            y_pred = self.forward(X)
            loss = self.MSEloss(y, y_pred)
            loss += reg * (self.linear.weights**2).sum()
            train_loss.append(loss)
            gradient = self.MSEloss.backward()
            self.backward(gradient)
            self.update(lr, reg)

            if log:
                if i % n == 0:
                    t_acc = ((y - y_pred)**2).mean()
                    y_val_pred = self.predict(X_val).squeeze()
                    v_acc = ((y_val - y_val_pred)**2).mean()
                    
                    train_acc.append(t_acc)                    
                    val_acc.append(v_acc)
                    print(f'iteration {i}: training loss {loss.mean():.2f} ' +
                           f' training accuracy {t_acc:.2f} val accuracy {v_acc:.2f}')


        return train_loss, train_acc, val_acc


class LogisticRegression:

    def __init__(self, in_dim, out_dim):
        """
        in_dim: number of input features
            if x.shape = [1000, 10], in_dim = 10
        out_dim: number of output features 
            out_dim = y.shape[1]
        """
        self.l1 = c.Linear(in_dim, out_dim)
        self.sigmoid = c.Sigmoid()
        self.BCEloss = mll.BinaryCrossEntropy()

    def forward(self, x):
        x = self.l1(x)
        y_pred = self.sigmoid(x)

        return y_pred

    def predict(self, x):
        x = self.l1.predict(x)
        y_pred = self.sigmoid.predict(x)

        return y_pred

    def backward(self, gradient):
        x = self.sigmoid.backward(gradient)
        self.l1.backward(x)

    def update(self, lr, reg):
        self.l1.update(lr, reg)

    def fit(self, X, y, lr, n_iter, reg=0, log=True, n=5, X_val=None, y_val=None):

        training_loss = []
        training_acc = []
        val_acc = []
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for i in range(n_iter):
            y_pred = self.forward(X)
            loss = self.BCEloss(y, y_pred)
            loss += reg * (self.l1.weights**2).sum()
            training_loss.append(loss)
            gradient = self.BCEloss.backward()
            self.backward(gradient)
            self.update(lr, reg)

            if log:
                if i % n == 0:
                    t_acc = (np.round(y_pred) == y).mean()
                    y_pred_val = self.predict(X_val).squeeze()
                    v_acc = (np.round(y_pred_val) == y_val).mean()
                    
                    training_acc.append(t_acc)
                    val_acc.append(v_acc)
                    print(f'iteration {i}: training loss {loss:.2f} ' +
                          f' training acc {t_acc:.2f} val acc {v_acc:.2f}')

        return training_loss, training_acc, val_acc
    
    
class TwoLayerNet:
    
    def __init__(self, in_dim, hidden_dim, out_dim, optim_rule='SGD', config={'rho': 0.}):
        self.linear_relu = c.LinearReLU(in_dim, hidden_dim, optim_rule, config)
        self.l2 = c.Linear(hidden_dim, out_dim, optim_rule, config)
        
    def forward(self, x):
        x = self.linear_relu.forward(x)
        out = self.l2(x)
        
        return out
    
    def predict(self, x):
        x = self.linear_relu.predict(x)
        out = self.l2.predict(x)
        
        return out
    
    def backward(self, grad):
        grad = self.l2.backward(grad)
        grad = self.linear_relu.backward(grad)
        
    def update(self, lr, reg):
        self.l2.update(lr=lr, reg=reg)
        self.linear_relu.update(lr=lr, reg=reg)
        
        
    def fit(self, x, y, x_val, y_val, loss_func: Callable, lr=0.01, reg=0., n_iter=500, n=5):
        
        training_loss = []
        train_acc = []
        val_acc = []
                
        for i in range(n_iter):
            y_pred = self.forward(x)
            loss = loss_func(y, y_pred)
            
            loss += reg * ((self.linear_relu.linear.weights**2).sum() + (self.l2.weights**2).sum())
            training_loss.append(loss)
            
            grad = loss_func.backward()
            self.backward(grad)
            self.update(lr, reg)
            
            if i % n == 0:
                t_acc = (np.argmax(y_pred, axis=1) == y).mean()
                train_acc.append(t_acc)
                v_acc = (np.argmax(self.predict(x_val), axis=1) == y_val).mean()
                val_acc.append(v_acc)
                
                print(f'Iteration {i}: loss {loss:.2f}, training acc {t_acc:.2f}, val acc {v_acc:.2f}')
                
        return training_loss, train_acc, val_acc
                
            
                
class ThreeLayerNet:
    
    def __init__(self, in_dim, hidden_dim, out_dim, optim_rule: str='SGD', config={'rho': 0.}, is_bn=True):
        self.is_bn = is_bn
        
        self.l1_relu = c.LinearReLU(in_dim, hidden_dim[0], optim_rule, config)
        self.l2 = c.Linear(hidden_dim[0], hidden_dim[1], optim_rule, config)
        if is_bn:
            self.bn = c.BatchNorm(hidden_dim[1])
        self.relu = c.ReLU()
        self.l3 = c.Linear(hidden_dim[1], out_dim, optim_rule, config)
        
    def forward(self, x):
        x = self.l1_relu.forward(x)
        x = self.l2(x)
        if self.is_bn:
            x = self.bn(x)
        x = self.relu(x)
        out = self.l3(x)
        
        return out
    
    def predict(self, x):
        x = self.l1_relu.predict(x)
        x = self.l2.predict(x)
        if self.is_bn:
            x = self.bn(x, mode='test')
        x = self.relu.predict(x)
        out = self.l3.predict(x)
        
        return out
    
    def backward(self, grad):
        grad = self.l3.backward(grad)
        grad = self.relu.backward(grad)
        if self.is_bn:
            grad = self.bn.backward(grad)
        grad = self.l2.backward(grad)
        grad = self.l1_relu.backward(grad)
        
    def update(self, lr, reg):
        self.l3.update(lr=lr, reg=reg)
        if self.is_bn:
            self.bn.update(lr=lr)
        self.l2.update(lr=lr, reg=reg)
        self.l1_relu.update(lr=lr, reg=reg)
        
        
    def fit(self, x, y, x_val, y_val, loss_func: Callable, lr=0.01, reg=0., n_iter=500, n=5):
        
        training_loss = []
        train_acc = []
        val_acc = []
                
        for i in range(n_iter):
            y_pred = self.forward(x)
            loss = loss_func(y, y_pred)
            
            loss += reg * ((self.l1_relu.linear.weights**2).sum() + (self.l2.weights**2).sum() + (self.l3.weights**2).sum())
            training_loss.append(loss)
            
            grad = loss_func.backward()
            self.backward(grad)
            self.update(lr, reg)
            
            if i % n == 0:
                t_acc = (np.argmax(y_pred, axis=1) == y).mean()
                train_acc.append(t_acc)
                v_acc = (np.argmax(self.predict(x_val), axis=1) == y_val).mean()
                val_acc.append(v_acc)
                
                print(f'Iteration {i}: loss {loss:.2f}, training acc {t_acc:.2f}, val acc {v_acc:.2f}')
                
        return training_loss, train_acc, val_acc        
            
            
            
            

            

            





            
            
            