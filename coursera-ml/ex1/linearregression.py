import math, sys, os
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
    """
    
    def __init__(self, alpha=0.01, iters=100, \
                    n_features=1, normalize=False):
        self.alpha = alpha
        self.n_iter = iters
        self.n_examples = 0
        self.n_features = n_features
        self.normalize = normalize
         
        self.theta = np.zeros(self.n_features+1)
        self.J_hist = np.zeros(self.n_iter) 
    
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
        J = (1/(2*float(self.n_examples))) * \
                np.sum(np.square((np.subtract(self.decision_function(X), y))))
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
                    np.dot(X.T, (np.subtract(self.decision_function(X), y)))
            
            #compute the updated cost
            J = self.compute_cost(X, y)
            
            self.J_hist[i] = J
            
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
        dataset on a [0-1] scale. The normalization formula
        is as follows:

        >>> x_j = (x_j - mu_j) / s_j

        where mu_j is the mean of a given feature
        and s_j is the standard deviation of the feature

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
            a = np.true_divide(np.subtract(xj, muj), sj)
            X_n[:,j] = a
        return X_n
    
    def normal_equations(self, X):
        """Compute the optimal parameters using
        the normal equations. This approach is the 
        following matrix multiplication 
        >>> theta = np.inv(X.T * X) (X.T * y)

        Parameters
        ----------
        X : numpy array-like matrix
            Training examples

        Returns
        -------
        theta : numpy array
            Optimal parameters
        """
        x_zero = np.ones(len(X))
        X_ne = np.column_stack([x_zero, X])
        theta = np.dot(np.linalg.inv(np.dot(X_ne.T, X_ne)), \
                                        np.dot(X_ne.T, y))
        return theta

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
            X = self.normalize_features(X) 

        x_zero = np.ones(len(X))
        X = np.column_stack([x_zero, X])
        
        self.compute_gradient(X, y)
    
    def predict(self, X_test):
        """Predicts continuous value for unseen test
        examples using the learned weights
        
        Parameters
        ----------
        X_text : numpy array-like matrix
            Test set examples
        
        """
        predictions = list()
        for x in X_test:
            y_pred = np.dot(x, self.theta.T)
            predictions.append(y_pred)
        return predictions
   
def load_data(input_file, delim=','):
    """Load dataset from file. Expects examples
    in the rows, features in the columns and 
    truth labels in the last column. 
    
    Parameters
    ----------
    input_file : str
        Path to input data file
        
    delim : str
        Type of delimiter for each field
        
    Returns
    -------
    X : numpy-like matrix
        Feature Matrix from input data
        
    y : numpy array
        Labels vector
    """
    
    data = np.loadtxt(input_file, delimiter=delim)
    n_features = data.shape[1]-1
    if n_features == 1:
        X = data[:,0]
    else:
        X = data[:,np.arange(n_features)]
    y = data[:,n_features]
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

def plot_alpha_cost_hist(alphas, alpha_cost_hist):
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost J')
    
    step = 5.0
    color = ['b', 'r', 'g', 'k', 'b--', 'r--', '--k', '--g']
    for i, J in enumerate(alpha_cost_hist):
        plt.plot(np.arange(len(J)), J, color[i], label='{0:0.3f}'
                 ''.format(alphas[i]));
        plt.xticks(np.arange(0, len(J)+step, step));    
    plt.legend(loc="lower right")
    plt.show() 

if __name__ == "__main__":
    
    #=============== For one variable =================== 
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
    print "Minimized Cost: %s" % regressor.J_hist[-1] 
    print "Optimal Theta: %s" % regressor.theta
    
    #apply learned weights to test data
    predictions = regressor.predict(X_test)
    for y_pred in predictions:
        print "Predicted: %.3f" % (y_pred * 10000)
    
    plot_decision(X, y, regressor)
    plot_gradient(X, regressor)
    
    #=============== For multiple variables =================== 
    #load training/test data
    input_file = os.path.abspath(\
                    os.path.join(os.path.dirname(".."), \
                        "ex1data2.txt"))
    X, y = load_data(input_file)
    X_test = np.array([[1, 1650, 3]])
  
    best_alpha = 0
    min_cost = sys.maxint
    best_theta = None
    alpha_cost_hist = list()
   
    alphas = [1.3, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001] 
    for i, alpha in enumerate(alphas):
        regressor = MyLinearRegressor(alpha=alpha, \
                    iters=50, n_features=2, normalize=True)
        
        regressor.fit(X,y)
        alpha_cost_hist.append(regressor.J_hist)
        cost = regressor.J_hist[-1]
        
        if cost <= min_cost:
            min_cost = cost
            best_alpha = alpha
            best_theta = regressor.theta
        
    print "\nOptimal alpha=%s\nOptimal theta=%s" % (best_alpha, best_theta)

    plot_alpha_cost_hist(alphas, alpha_cost_hist)

    #normalize each input feature in test example
    X_test = np.array([[1, \
                np.true_divide(\
                    np.subtract(1650, np.mean(X[:,0])), \
                        np.std(X[:,0])), \
                
                np.true_divide(\
                    np.subtract(3, np.mean(X[:,1])), \
                        np.std(X[:,1])) \
                ]])
    
    price = np.dot(X_test, best_theta.T)
    print "Predicted $%.0f" % price

    #gradient minimization using the normal equations
    theta = regressor.normal_equations(X)
    x_test = np.array([[1, 1650, 3]])
    price = np.dot(x_test, theta.T) 

    print "\nOptimal theta: %s" % theta
    print "Predicted Price: $%.0f" % price

