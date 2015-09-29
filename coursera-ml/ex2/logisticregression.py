import sys
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score


class MyLogisticRegression(object):
    def sigmoid_hypothesis(self, z):
        """The decision function used for prediction

        Parameters
        ----------
        z : numpy array-like matrix
            Linear regression hypothesis applied to training examples

            h(X) = theta.T.dot(X)

        Returns
        -------
        float
            Predicted value
        """
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, theta, X, y):
        """Given a set of training examples, predictions, and parameters
        estimate the cost using the sigmoid hypothesis function

        Parameters
        ----------
        theta : numpy array-like matrix
            Parameters of the model

        X : numpy array-like matrix
            Training examples

        y : numpy array
            Labels vector

        Returns
        -------
        float
            Cost of the predictions under given parameters
        """
        m = X.shape[0]
        g = self.sigmoid_hypothesis(X.dot(theta))
        return (-y.T.dot(np.log(g)) -
                (1 - y).T.dot(np.log(1 - g))) / float(m)

    def compute_gradient(self, theta, X, y):
        """Compute the gradient give a set of parameters

        Parameters
        ----------
        theta : numpy array-like matrix
            Parameters to be learned by the model

        X : numpy array-like matrix
            Training examples

        y : numpy array
            Labels vector

        Returns
        -------
        float
            gradient under some set of parameters
        """
        g = self.sigmoid_hypothesis(X.dot(theta))
        loss = np.subtract(g, y)
        return X.T.dot(loss) / float(X.shape[0])

    def gradient_descent(self, theta, X, y, alpha=4.8, max_iter=400):
        """Determines the optimal parameters for the model using gradient descent

        Parameters
        ----------
        theta : numpy array-like matrix
            Parameters to be learned by the model

        X : numpy array-like matrix
            Training examples

        y : numpy array
            Labels vector

        alpha : float
            The learning rate

        max_iter : int
            Max number of iterations for training

        Returns
        -------
        tuple
            (Learned parameters, minimal cost, cost history, gradient history)
        """
        best_theta = theta
        min_cost = sys.maxint
        J_hist = np.zeros(max_iter)
        grad_hist = list()

        for i in np.arange(max_iter):
            grad = self.compute_gradient(theta, X, y)
            grad_hist.append(grad)
            theta -= (alpha * grad)

            J = self.compute_cost(theta, X, y)
            J_hist[i] = J
            if J < min_cost:
                min_cost = J
                best_theta = theta
        return best_theta, min_cost, J_hist, grad_hist

    def fminunc_optimizer(self, theta, X, y):
        """Determines the optimal parameters for the model using an advanced solver

        Parameters
        ----------
        theta : numpy array-like matrix
            Parameters to be learned by the model

        X : numpy array-like matrix
            Training examples

        y : numpy array
            Labels vector

        Returns
        -------
        tuple
            (Learned parameters, minimal cost, _, _)
        """
        options = {'full_output': True, 'maxiter': 400}
        theta, cost, _, _, _ = \
            op.fmin(lambda t: self.compute_cost(t, X, y), theta, **options)

        return theta, cost, _, _

    def find_learning_rate(self, theta, X, y, tol=1e-12, step=0.1):
        """Find the optimal learning to train the model

        theta : numpy array-like matrix
            Parameters to be learned by the model

        X : numpy array-like matrix
            Training examples

        y : numpy array
            Labels vector

        tol : float
            Minimum difference in cost between successive iterations

        step : float
            The amount by which the learning rate will be decremented

        Returns
        -------
        float
            The optimal learning rate
        """
        alphas = np.arange(0, 10, step)
        alphas_hist = np.zeros((alphas.shape[0], 1))

        learning_rate = sys.maxint
        theta = np.zeros((X.shape[1], 1))
        found = False

        for i, alpha_i in enumerate(alphas):
            theta, cost, _, _ = self.gradient_descent(theta, X, y, alpha_i)
            if i == 0:
                t = cost
            if i > 0 and t - cost < tol and not found:
                learning_rate = alpha_i
                found = True
            alphas_hist[i] = cost
            t = cost
        return learning_rate

    def normalize_features(self, X):
        """This function normalizes each feature in the
        dataset on a [0-1] scale. The normalization formula
        is as follows:
        >>> x_j = (xj - muj) / sj
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
        mu = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mu) / std
        return X, mu, std

    def fit(self, X, y, solver='bgd'):
        """Determines the optimal parameters for the model using a solver

        Parameters
        ----------
        X : numpy array-like matrix
            Training examples

        y : numpy array
            Labels vector

        solver : str
            Specified the type of solver to use.
            Choose from 1) batch gradient descent and 2) fminunc

        Returns
        -------
        numpy array-like matrix
            Learned parameters
        """
        X, _, _ = self.normalize_features(X)
        X = np.column_stack([np.ones(len(X)), X])
        y = y.reshape(X.shape[0], 1)

        theta = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
        J = self.compute_cost(theta, X, y)
        print("Initial Cost: %.3f" % J)

        grad = self.compute_gradient(theta, X, y)
        print("Initial Gradient: %s" % grad.T)

        if solver == 'bgd':
            print "Computing the learning rate..."
            alpha = self.find_learning_rate(theta, X, y)
            print "Best alpha: %.1f" % alpha
            return self.gradient_descent(theta, X, y, alpha)
        elif solver == 'fminunc':
            return self.fminunc_optimizer(theta, X, y);

    def predict(self, theta, X):
        """Predicts discrete boolean values for test
        examples using the learned weights

        Parameters
        ----------
        theta : numpy array
            Learned parameters

        X_test : numpy array-like matrix
            Test set examples

        Returns
        -------
        numpy array
            Predictions for test set examples
        """
        X, _, _ = clf.normalize_features(X)
        X = np.column_stack([np.ones(len(X)), X])

        y_pred = np.zeros((X.shape[0], 1))
        h = X.dot(theta)
        g = self.sigmoid_hypothesis(h)
        x_pos = np.where(g >= 0.5)
        y_pred[x_pos] = 1
        return y_pred


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
    n_features = data.shape[1] - 1
    if n_features == 1:
        X = data[:, 0]
    else:
        X = data[:, np.arange(n_features)]
    y = data[:, n_features]
    assert len(X) == len(y)
    return X, y


def plot_data(X, y):
    plt.title('Scatter plot of training data')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 score')

    x_pos = np.where(y == 1)
    x_neg = np.where(y == 0)

    adm = plt.scatter(X[x_pos][:, 0], X[x_pos][:, 1], marker='+', color='k')
    not_adm = plt.scatter(X[x_neg][:, 0], X[x_neg][:, 1], marker='o', color='y')
    plt.legend((adm, not_adm), ('Admitted', 'Not Admitted'), scatterpoints=1, loc="upper right")
    plt.xlim(30, 100)
    plt.ylim(30, 100)
    plt.show()


def plot_decision_boundary(theta, X, y, clf):
    plt.title('Scatter plot of training data')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 score')

    X, _, _ = clf.normalize_features(X)
    x_pos = np.where(y == 1)
    x_neg = np.where(y == 0)

    plot_x = np.array([min(X[:, 0]) - 2, max(X[:, 0]) + 2])
    plot_y = (-1 / theta[2]) * ((theta[1] * plot_x) + theta[0])

    adm = plt.scatter(X[x_pos][:, 0], X[x_pos][:, 1], marker='+', color='k')
    not_adm = plt.scatter(X[x_neg][:, 0], X[x_neg][:, 1], marker='o', color='y')
    plt.legend((adm, not_adm), ('Admitted', 'Not Admitted'), scatterpoints=1, loc="upper right")
    plt.plot(plot_x, plot_y, '-b')
    plt.show()


if __name__ == '__main__':
    clf = MyLogisticRegression()

    X, y = load_data("ex2data1.txt")

    print("Number of training examples: %i" % X.shape[0])
    print("Number of features: %s" % X.shape[1])

    plot_data(X, y)

    # =============== gradient descent =======================
    theta, min_cost, J_hist, grad_hist = clf.fit(X, y)
    print("Minimum gradient %s" % theta.T)
    print("Minimum cost %s" % min_cost)

    X_test = np.array([1,
                       np.true_divide(np.subtract(45, np.mean(X[:, 0])), np.std(X[:, 0])),
                       np.true_divide(np.subtract(85, np.mean(X[:, 1])), np.std(X[:, 1]))])

    z = X_test.dot(theta)
    print("Probability of Admittance %s" % clf.sigmoid_hypothesis(z))

    y_pred = clf.predict(theta, X)
    print("Precision: %.2f\n" % precision_score(y, y_pred))
    plot_decision_boundary(theta, X, y, clf)

    # =============== advanced optimization =======================
    theta, min_cost, J_hist, grad_hist = clf.fit(X, y, 'fminunc')
    print("\nMinimum gradient %s" % theta.T)
    print("Minimum cost %s" % min_cost)
    z = X_test.dot(theta)
    print("Probability of Admittance %s" % clf.sigmoid_hypothesis(z))
