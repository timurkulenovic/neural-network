import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import StandardScaler


CLS = "classification"
REG = "regression"


class ANN:
    def __init__(self, units, lambda_, seed, num, save_grad, type_):
        """
        :param units: list
        :param lambda_: float
        :param seed: float
        :param num: Bool - whether to use numerical gradient or not
        :param save_grad: Bool - whether to save gradients or not
        """
        self.type_ = type_
        self.units = units
        self.lambda_ = lambda_
        self.seed = seed
        self.num = num
        self.X, self.y = None, None
        self.sizes = None
        self.A = None
        self.Z = None
        self.Y, self.y_map = None, None
        self.save_grad = save_grad
        self.grads = {"backprop": [], "numerical": []}

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.X = X
        self.y = y
        if self.type_ == CLS:
            self.y_map = {y_i: i for i, y_i in enumerate(np.unique(y))}
            y_int = np.array([self.y_map[y_i] for y_i in self.y])
            self.Y = (np.arange(np.max(y_int) + 1) == y_int[:, None]).astype(float)
        self.sizes = self.w_matrix_sizes(self.X.shape[1], self.Y.shape[1] if self.type_ == CLS else 1)
        init_weights = np.round(np.random.rand(sum([r * c for r, c in self.sizes])), 1)
        opt = fmin_l_bfgs_b(func=self.cost, fprime=self.grad if not self.num else self.num_grad, x0=init_weights, maxiter=200)
        W, b = self.unflatten(opt[0])
        return ANNModel(W, b, self.type_, self.y_map)

    def w_matrix_sizes(self, n, j):
        if len(self.units) == 0:
            return [(n + 1, j)]
        sizes = [(n + 1, self.units[0])]
        for i in range(1, len(self.units)):
            sizes.append((self.units[i-1] + 1, self.units[i]))
        sizes.append((self.units[-1] + 1, j))
        return sizes

    def unflatten(self, weights):
        """
        :param weights: np.array
        :return: pair W, b where each is list of arrays
        """
        splits = np.cumsum([r * c for r, c in self.sizes])
        W, b = [], []
        for s, (r, c) in zip(np.split(weights, splits), self.sizes):
            W.append(s[:-c].reshape((r - 1, c)))
            b.append(s[-c:])
        return W, b

    def grad(self, weights):
        W, b = self.unflatten(weights)
        grad, dE_ds = [], {}
        for L in list(range(1, len(self.A)))[::-1]:
            grad_L = []
            for j in range(W[L-1].shape[0]):
                dE_dA = None if L < len(W) else self.A[L] - self.Y if self.type_ == CLS else 2 * (self.A[L] - self.y.reshape(-1, 1))
                dE_ds[L] = dE_ds.get(L, dE_dA if L == len(W) else (dE_ds[L + 1].dot(W[L].T) * (self.A[L] * (1 - self.A[L]))))
                grad_L.append((dE_ds[L].T * self.A[L-1][:, j]).T)
            grad_L.append(dE_ds[L])     # for biases
            grad[:0] = np.sum(grad_L, axis=1).reshape(-1)
        reg = np.array([i for sub in [[1] * ((r - 1) * c) + c * [0] for r, c in self.sizes] for i in sub], dtype=np.float32)
        grad = np.array((1 / len(self.X)) * (np.array(grad) + self.lambda_ * reg * weights))
        if self.save_grad:
            self.grads["backprop"].append(grad)
            self.grads["numerical"].append(self.num_grad(weights))
        return grad

    def num_grad(self, weights):
        eps = 10 ** -6
        grad = []
        for i in range(len(weights)):
            eh = np.zeros(len(weights))
            eh[i] = eps
            grad.append((self.cost(weights + eh, False) - self.cost(weights - eh, False)) / (2 * eh[i]))
        return np.array(grad)

    def cost(self, weights, set_params=True):
        eps = 1e-15
        W, b = self.unflatten(weights)
        A, Z = [self.X], []
        for i in range(len(W)):
            Z.append(A[-1].dot(W[i]) + b[i])
            A.append(sig(Z[-1]) if i < len(W) - 1 else (softmax_matrix(Z[-1]) if self.type_ == CLS else Z[-1]))
        if set_params:
            self.A, self.Z = A, Z
        reg = self.lambda_ * sum([np.sum(W_i ** 2) for W_i in W])
        cost = np.sum(-np.log(A[-1] + eps) * self.Y) if self.type_ == CLS else ((A[-1].reshape(-1) - self.y) ** 2).sum()
        J = 1 / len(self.X) * (cost + 1 / 2 * reg)
        return J

    def get_grads(self):
        return self.grads


class ANNModel:
    def __init__(self, W, b, type_, y_map):
        self.W = W
        self.b = b
        self.type_ = type_
        self.y_map = y_map

    def get_y_map(self):
        return self.y_map

    def predict(self, X):
        A = X
        for i in range(len(self.W)):
            Z = A.dot(self.W[i]) + self.b[i]
            A = sig(Z) if i < len(self.W) - 1 else (softmax_matrix(Z) if self.type_ == CLS else Z)
        return A if self.type_ == CLS else A.reshape(-1)

    def weights(self):
        return [np.append(W, [b], axis=0) for W, b in zip(self.W, self.b)]


class ANNClassification:
    def __init__(self, units, lambda_, seed=0, num=False, save_grad=False):
        """
        :param units: list
        :param lambda_: float
        :param seed: float
        :param num: Bool - whether to use numerical gradient or not
        """
        self.learner = ANN(units, lambda_, seed, num, save_grad, type_=CLS)

    def fit(self, X, y):
        return self.learner.fit(X, y)


class ANNRegression:
    def __init__(self, units, lambda_, seed=0, num=False, save_grad=False):
        """
        :param units: list
        :param lambda_: float
        :param seed: float
        :param num: Bool - whether to use numerical gradient or not
        """
        self.learner = ANN(units, lambda_, seed, num, save_grad, type_=REG)

    def fit(self, X, y):
        return self.learner.fit(X, y)


def norm(x):
    return np.sqrt(x.dot(x))


def softmax_matrix(m):
    return np.array([softmax(m_i) for m_i in m])


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)


def sig(x):
    return 1 / (1 + np.exp(-x))


def log_loss(predicted, y):
    return sum([-np.log(p_i[t_i]) for p_i, t_i in zip(predicted, y.astype(int))]) / len(predicted)



if __name__ == '__main__':
    # Example
    X_ = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
    y_ = np.array([0, 1, 1, 0])

    fitter = ANNClassification(units=[2], lambda_=0.0005, num=False, save_grad=True)
    m = fitter.fit(X_, y_)
    pred = m.predict(X_)
    print(pred)
