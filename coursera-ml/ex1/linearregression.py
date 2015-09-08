import os
import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegressor(object):
    """Linear Regression for n-variables
    
    Parameters
    ----------
    alpha : float 
        (optional) Learning rate. Default=0.01
        
    iters : int
        (optional) Max iterations for gradient descent. 
        Default=100
        
    theta : numpy array-like matrix
        Parameters to be learned
        
    n_features : int
        (optional) Number of features in training set
        Default=1
    
    normalize: boolean
        (optional) Whether features should be normalized     
        Default=False
    
    Attributes
    ----------
    n_examples : int
        Number of training examples 

    theta : numpy array-like matrix
        Parameters to be learned by the regressor

    J_hist : numpy array
        Cost function output at each iteration        

    theta_hist : numpy array-like matrix
        Learned parameters at each iteration
        
    """
    
    def __init__(self, alpha=0.01, iters=100, \
                    n_features=1, normalize=False):
        self.alpha = alpha
        self.n_iter = iters
        self.n_examples = 0
        self.n_features = n_features
        self.normalize = normalize
         
        self.theta = None
        self.J_hist = np.zeros(self.n_iter) 
        self.theta_hist = np.zeros((self.n_iter,n_features+1)) 
    
    def compute_cost(self, X, y):
        """Given a set of training examples, predictions, and parameters
        estimate the cost using the linear hypothesis function
        
        Parameters
        ----------
        X : numpy array-like matrix 
            Training examples
            
        y : numpy array
            Labels vector
            
        Returns
        -------
        J : float
            Cost of the predictions under given parameters
        """
        J = (1/float(2*self.n_examples)) * \
                np.sum(np.square(self.decision_function(X) - y))
        return J
    
    def compute_gradient(self, X, y):
        """The function updates the paramaters using
        batch gradient descent. The default stopping
        condition is a fixed number of iterations. 
        
        Parameters
        ----------
        X : numpy array-like matrix 
            Training examples
            
        y : numpy array
            Labels vector
        """
        
        for i in np.arange(self.n_iter):
            #update the weights
            self.theta = self.theta - \
                            (self.alpha/float(self.n_examples)) * \
                                np.dot(X.T, (self.decision_function(X) - y))
            
            #compute the updated cost
            J = self.compute_cost(X, y)
            
            self.J_hist[i] = J
            self.theta_hist[i] = self.theta
            
    def get_params(self):
        """Retrieves the weights of the trained model"""
        return self.theta
    
    def decision_function(self, X):
        """The decision function used for prediction
        
        Parameters
        ----------
        X : numpy array-like matrix 
            Training examples
        
        Returns
        -------
        h : float
            Predicted values
        """
        return np.dot(X, self.theta.T)
    
    def normalize_features(self, X):
        """This function normalizes each feature in the
        dataset on a [0-1] scale

        Parameters
        ----------
        X : numpy array-like matrix 
            Training examples

        Returns
        -------
        X_n : numpy array-like matrix
            Normalized training examples
        """
        n_ex = X.shape[0]
        n_fts = X.shape[1]
        X_n = np.zeros((n_ex, n_fts))
        for j in np.arange(n_fts):
            xj = X[:,j]
            muj = np.mean(xj)
            sj = np.std(xj) 
            a = np.true_divide(\
                        np.subtract(xj, muj), \
                            sj)
            X_n[:,j] = a
        return X_n

    def fit(self, X, y):
        """Determines the optimal weights/parameters
        for the model by using batch gradient descent 
        
        Parameters
        ----------
        X : numpy array-like matrix 
            Training examples
            
        y : numpy array
            Labels vector
        """
        self.n_examples = len(X)
        
        if self.normalize:
            print "Normalizing Features..."
            X = self.normalize_features(X) 

        x_zero = np.ones(len(X))
        X = np.column_stack([x_zero, X])
              
        n = X.shape[1] 
        self.theta = np.zeros(n)
        
        J = self.compute_cost(X, y)
        print "Initial Cost: %.2f" % J
        
        self.compute_gradient(X, y)
        
        print "Minimized Cost: %.2f" % self.J_hist[-1]
        print "Optimal Thetas: %s" % self.theta
    
    def predict(self, X_test):
        """Predicts continuous value for unseen test
        examples using the learned weights
        
        Parameters
        ----------
        X_text : numpy array-like matrix
            Test set examples
        
        """
        for x in X_test:
            y_pred = np.dot(x, self.theta.T)
            print "Predicted: %.0f" % y_pred
    
def load_data(input_file, delim=','):
    """Loads the data for our problem"""
    
    data = np.loadtxt(input_file, delimiter=delim)
    X = data[:, 0]
    y = data[:, 1]
    assert len(X) == len(y)
    return X, y

def plot_decision(X, y, regressor):
    plt.title('Distribution of Profits by Population') 
    plt.xlabel('City Population (in 10,000s)')
    plt.ylabel('Profits (in $10,000s)')
    
    theta = regressor.get_params()
    step = 2.0
    x_zero = np.ones(len(X))
    X_n = np.column_stack([x_zero, X])
 
    x_ticks = (np.ceil(X))
    plt.xticks(np.arange(min(x_ticks)-step, max(x_ticks)+step, step))
    plt.xlim(0, max(x_ticks)+step)
    plt.ylim(0, max(y)+step)
    plt.scatter(X, y, s=50, marker='x', c='r');
    plt.plot(X, np.dot(X_n, theta.T), '--b');
    plt.plot([0,max(X)], [1,max(X)], '--k');
    plt.show()

def plot_gradient(X, regressor):
    plt.title('Gradient Distribution') 
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost J')
    
    J_hist = regressor.J_hist
    
    step = 0.5
    plt.yticks(np.arange(min(J_hist)-step, max(J_hist)+step, step))
    plt.plot(np.arange(len(J_hist)), J_hist, '.g', linewidth=0.2)
    plt.show()
    
if __name__ == "__main__":
    input_file = os.path.abspath(\
                    os.path.join(os.path.dirname(".."), \
                        "ex1data1.txt"))
    
    #load training/test data
    X, y = load_data(input_file)
    X_test = np.array([[1, 3.5], [1, 7.0]])
    
    regressor = MyLinearRegressor(alpha=0.01, \
                            iters=1500, n_features=1)
    #learn the weights
    regressor.fit(X,y)
    
    #apply learned weights to test data
    regressor.predict(X_test)
    
    plot_decision(X, y, regressor)
    plot_gradient(X, regressor)

    #load data for multiple variable case
    
