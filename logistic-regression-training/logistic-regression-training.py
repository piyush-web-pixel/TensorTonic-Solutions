import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w=None
    b=None
    X=np.insert(X,0,1,axis=1)
    weights=np.ones(X.shape[1])
    for i in range(steps):
        y_hat=_sigmoid(np.dot(X,weights))
        weights=weights+lr*(np.dot((y-y_hat),X)/X.shape[0])

    w=weights[1:]
    b=weights[0]

    return w,b
    
                
