# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#import scipy
from scipy.special import expit
from scipy.stats import poisson
get_ipython().run_line_magic('matplotlib', 'inline')

#BayesClassifier
#loading data
X = np.genfromtxt('X.csv', delimiter=',')
y = np.genfromtxt('y.csv', delimiter=',')

y = np.reshape(y, (4600, 1))

def getLambda(X,y):
    lamdba_0_d = (np.sum(X * (y != 1), axis = 0) + 1) / (np.sum(y != 1 ) + 1)
    lamdba_1_d = (np.sum(X * (y == 1), axis = 0) + 1) / (np.sum(y == 1 ) + 1)

    return lamdba_0_d, lamdba_1_d

#print(getLambda(X, y))

def get_prior(y):
    pi_0 = np.sum(y != 1) / len(y)
    pi_1 = np.sum(y == 1) / len(y)
    return pi_0, pi_1

print(get_prior(y))

def BayesClassifier(X, y, X_test):
    #training phase
    pi_0, pi_1 = get_prior(y)
    lambda_0_d, lambda_1_d = getLambda(X,y)
    #prediction on new test set
# Compute the log sum of the probabilities for each dimension for class 0 (non spam)
    log_prob_0 = np.log(pi_0) + np.sum(np.log(lambda_0_d) * X_test - lambda_0_d, axis=1)
# Compute the log sum of the probabilities for each dimension for class 1 (spam)
    log_prob_1 = np.log(pi_1) + np.sum(np.log(lambda_1_d) * X_test - lambda_1_d, axis=1)
    return (log_prob_0 < log_prob_1).astype(int)

kf=KFold(n_splits=10, random_state=None, shuffle=False)
y_prediction = []
labels = []

for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_pred = BayesClassifier(X_train, y_train, X_test)

    y_prediction.append(y_pred)
    labels.append(y_test)

y_prediction = np.concatenate(y_prediction)
y_prediction

labels = np.concatenate(labels)

confusion_matrix(labels, y_prediction)

accuracy_score(labels, y_prediction)

ax= plt.subplot()
conf_mat = confusion_matrix(labels, y_prediction);
#conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_mat, annot=True, fmt = "g", cmap = 'Blues');
ax.set_xlabel('labels', fontsize=12)
ax.xaxis.set_ticklabels(['spam', 'ham'], fontsize = 15)
ax.set_ylabel('Predictions', fontsize=12)
ax.yaxis.set_ticklabels(['spam', 'ham'], fontsize = 15)
plt.figure(figsize = (15,10));

plt.savefig("cm.png")

y = np.reshape(y, (4600, 1))


lamdba_0_d = (np.sum(X * (y != 1), axis = 0) + 1) / (np.sum(y != 1 ) + 1)
lamdba_1_d = (np.sum(X * (y == 1), axis = 0) + 1) / (np.sum(y == 1 ) + 1)

plt.stem([i for i in range(1, 55)], lamdba_1_d)
plt.stem([i for i in range(1, 55)], lamdba_0_d, linefmt='r', markerfmt='r*')
plt.show()

lamdba_0_d.shape

#-------------------------------------------------------------------------------
#Logistic Regression
X = np.genfromtxt('X.csv', delimiter=',')
y = np.genfromtxt('y.csv',delimiter=',')

y = np.reshape(y, (4600, 1))

def append_column_one(data):
    append_ones = np.ones((data.shape[0],1))
    data = np.hstack((append_ones, data))
    return data

X = append_column_one(X)
y[y==0] = -1

def calculate_sigmoid(X):
    return expit(X)

def calculate_update(X, y, w, sigmoid_iter):
    update = np.zeros(len(w))
    for i in range(0, X_train.shape[0]):
        update += y[i] * (1 - sigmoid_iter[i]) * X[i]
    return update

def calculate_objectiveFunc(X, y, w):
    output = 0
    sigmoid_iter = []
    for i in range(0, X.shape[0]):
        sigmoid_value = calculate_sigmoid(y[i] * np.dot(X[i], w))
        sigmoid_iter.append(sigmoid_value)
        output += np.log(sigmoid_value)
    return output, sigmoid_iter

objective_value = []
w = np.zeros(X.shape[1])
for t in range(1, 1001):
    #print "Iteration ", t
    learning_rate = 0.01 / 4600
    iter_objectiveVal, sigmoid_iter = calculate_objectiveFunc(X, y, w)
    objective_value.append(iter_objectiveVal)
    #print iter_objectiveVal
    w = w + (learning_rate * calculate_update(X, y, w, sigmoid_iter))

X.shape
y.shape

kf=KFold(n_splits=10, random_state=None, shuffle=False)
objectives = []
run_num = 1

for train_index, test_index in kf.split(X_train):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    objective_value = []
    w = np.zeros(X_train.shape[1])
    for t in range(1, 1001):
        learning_rate = 0.01 / 4600
        iter_objectiveVal, sigmoid_iter = calculate_objectiveFunc(X_train, y_train, w)
        objectives.append((run_num,iter_objectiveVal[0], t))
        w = w + (learning_rate * calculate_update(X_train, y_train, w, sigmoid_iter))
        #print(iter_objectiveVal)
    run_num += 1

    kf=KFold(n_splits=10, random_state=None, shuffle=False)

obj = pd.DataFrame(objectives, columns = ['run_num','learning_objectives', 'iter_num'])
obj.run_num = obj.run_num.astype(str)

sns.lineplot(data = obj, x = "iter_num", y = "learning_objectives", hue = "run_num")
plt.title("Logistic Regression (Steepest Ascent)")

#--------------------------------------------------------------------------------
#Newton's Method
X = np.genfromtxt('X.csv', delimiter=',')
y = np.genfromtxt('y.csv')

y = np.reshape(y, (4600, 1))

def append_column_one(data):
    append_ones = np.ones((data.shape[0],1))
    data = np.hstack((append_ones, data))
    return data

X = append_column_one(X)
y[y==0] = -1

def calculate_deltaupdate(X, y, weights):
    update = np.zeros(len(weights))
    for i in range(0, X.shape[0]):
        update = update + y[i] * (1 - calculate_sigmoid(y[i] * np.dot(X[i], weights))) * X[i]
    return update

def calc_squareinvupdate(X, y, weights):
    output_matrix = np.zeros((len(weights), len(weights)))
    for i in range(0, X.shape[0]):
        sig_value = calculate_sigmoid(np.dot(X[i],weights))
        output_matrix += sig_value * ( 1 - sig_value)*np.outer(X[i], X[i])
    return np.linalg.inv(-output_matrix)

def calculate_objectiveFunc(X, y, weights):
    #print(X[0].shape)
    output = 0
    for i in range(0, X.shape[0]):
        #print(calculate_sigmoid(y[i] * np.dot(X[i], weights)))
        output += np.log(calculate_sigmoid(y[i] * np.dot(X[i], weights)))
    return output

objective_value = []
w = np.zeros(X.shape[1])
#print len(w)
for t in range(1, 101):
    #print "Iteration ", t
    learning_rate = 1
    #print "Objective function value ", calculate_objectiveFunc(X_train, y_train, w)
    objective_value.append(calculate_objectiveFunc(X, y, w))
    w = w - (learning_rate * np.dot(calc_squareinvupdate(X, y, w),
                                    calculate_deltaupdate(X, y, w)))

kf=KFold(n_splits=10, random_state=None, shuffle=False)

X_train_all = []
y_train_all = []
score = []
models=[]
confusion_matrix = []

objectives = []
run_num = 1

for train_index, test_index in kf.split(X):
#     print(train_index)
#     print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    objective_value = []
    w = np.zeros(X_train.shape[1])
    for t in range(1, 101):
        #\print(t)
        learning_rate = 1
        iter_objectiveVal = calculate_objectiveFunc(X_train, y_train, w)
        objectives.append((run_num,iter_objectiveVal[0], t))
        w = w - (learning_rate * np.dot(calc_squareinvupdate(X_train, y_train, w),
                                    calculate_deltaupdate(X_train, y_train, w)))
    run_num += 1

obj = pd.DataFrame(objectives, columns = ['run_num','learning_objectives', 'iter_num'])
obj.run_num = obj.run_num.astype(str)

sns.lineplot(data = obj, x = "iter_num", y = "learning_objectives", hue = "run_num")
plt.title("Newthon Method")

#----------------------------------------------------------------------------------
#Gaussian Process:

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import pyplot as pl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D

from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from numpy.linalg import inv

X_train = np.genfromtxt('X_train.csv', delimiter=',')
y_train = np.genfromtxt('y_train.csv',delimiter=',')
X_test = np.genfromtxt('X_test.csv', delimiter=',')
y_test = np.genfromtxt('Y_test.csv', delimiter=',')

class gaussian(object):

    def __init__(self,X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def init_kernel(self, b, var):
        self.b = b
        self.var = var
        self.Kernel_train = self.get_kernel_matrix(self.X_train)
        self.Kernel_test = self.get_kernel_matrix(self.X_test)

    def calculate_kernel(self, x):
        def rbf(x1,x2):
            x1 = np.asarray(x1).flatten()
            x2 = np.asarray(x2).flatten()
            return np.exp(-1/self.b*np.dot(x1-x2,x1-x2))
        return np.apply_along_axis(rbf, 1,self.X_train,x)

    def get_kernel_matrix(self, X):
        row = X.shape[0]
        K = np.zeros(shape = (row, self.X_train.shape[0]))
        for i in range(row):
            K[i,:] = self.calculate_kernel(X[i,:])
        return K

    def predict_test(self):
        self.test_prediction = self.Kernel_test.dot(np.linalg.inv(
                                    self.var*np.identity(self.Kernel_train.shape[0]) + self.Kernel_train
                                    )).dot(self.y_train)

    def rmse(self):
        x1 = np.asarray(self.y_test).flatten()
        x2 = np.asarray(self.test_prediction).flatten()
        self.test_rmse = np.sqrt(np.dot(x1-x2, x1-x2)/self.X_test.shape[0])

def rmse_values(X_train, y_train, X_test,y_test):
    gp = gaussian(X_train, y_train, X_test,y_test)

    b_vals = [5,7,9,11,13,15]
    sigma_sqrd = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

    result_matrix = pd.DataFrame(index = b_vals, columns = sigma_sqrd)

    for b in b_vals:
        for var in sigma_sqrd:
            gp.init_kernel(b, var)
            gp.predict_test()
            gp.rmse()
            result_matrix.loc[b, sigma_sqrd] = gp.test_rmse
    print(result_matrix)
    result_matrix.to_csv('hw2_resultMatrix.csv')


#loading hw1 data
X_train = np.genfromtxt('hw1_X_train.csv', delimiter=',')
X_test = np.genfromtxt('hw1_X_test.csv', delimiter=',')
Y_train = np.genfromtxt('hw1_y_train.csv',delimiter=',')
Y_test = np.genfromtxt('hw1_Y_test.csv', delimiter=',')
Y_train = Y_train.reshape(350,1)
Y_test = Y_test.reshape(42,1)
assert np.shape(X_train) == (350, 7)
assert np.shape(X_test) == (42, 7)
assert np.shape(Y_train) == (350, 1)
assert np.shape(Y_test) == (42, 1)

rmse_values(X_train, Y_train, X_test,Y_test)

def dimension_4th(X_train, Y_train,X_test, Y_test):
    gp = gaussian(X_train[:,3].reshape((350,1)),Y_train,X_train[:,3].reshape((350,1)),_train)
    gp.init_kernel(b = 4, var = 2)
    gp.predict_test()

plt.scatter(X_train[:,3], Y_train)
variable = X_train[:,3].reshape((350,1))
f = interp1d(variable.flatten(), Y_train.flatten())
xnew = np.linspace(-1.5, 1.9, num=3.5, endpoint=True)

plt.plot(xnew, f(xnew), color='red')
plt.title("4th Dimension Visualization")
plt.xlabel("Car Weight");
