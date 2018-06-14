import numpy as np
# strange bug for mac matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv

def loadIrisData(filename):
    Y = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        count = 0
        for row in spamreader:
            arr = row[0].split(',')
            if (arr[4] == 'Iris-setosa'):
                Y.append(1.0)
            else:
                Y.append(0.0)
            # if (arr[4] == 'Iris-virginica'):
            #     Y.append(3)
            if (count == 0):
                X = np.matrix([list(map(float, arr[0:4]))])
            else:
                X = np.r_[X, [list(map(float, arr[0:4]))]]
            count += 1
    return np.squeeze(np.asarray(X.transpose())), np.squeeze(np.asarray(np.matrix([Y]).transpose()))

# initParameter
def initParameter(inputSize, outputSize):
    params = {}
    params["W"] = np.zeros((inputSize, outputSize), dtype=float)
    params["b"] = np.zeros((outputSize, 1), dtype=float)
    return params

# computeLoss
def computeLoss(Yhat, Y):
    return -Y*np.log(Yhat)-(1-Y)*np.log(1-Yhat)

# mean sum of loss
def computeCost(Yhat, Y):
    loss = computeLoss(Yhat, Y)
    return np.sum(loss, axis=0, keepdims=True)/Yhat.shape[0]

# computeGradient
def computeGradient(X, Y, Yhat):
    grad = {}
    dZ = Yhat - Y
    grad["dW"] = np.dot(X, dZ)
    grad["db"] = np.sum(dZ, axis=0, keepdims=True)/dZ.shape[0]
    return grad

# updateParameters
def updateParameters(parameters, grad, learningRate):
    parameters["W"] = parameters["W"] - learningRate * grad["dW"]
    parameters["b"] = parameters["b"] - learningRate * grad["db"]
    return parameters

# sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

# trainLR
def trainLR(X, Y, parameters, iterations, learningRate):
    costs = []
    for i in range(0, iterations):
        W = parameters["W"]
        b = parameters["b"]
        Yhat = sigmoid(np.dot(W.transpose(), X) + b).transpose()
        cost = computeCost(Yhat, Y)
        costs.append(cost)
        print("cost of iteration {} is {}".format(i, cost))
        grad = computeGradient(X, Y, Yhat)
        parameters = updateParameters(parameters, grad, learningRate)
    return parameters, costs

# predict
def predict(X, parameters, Y):
    Ypredict = np.round(sigmoid(np.dot(parameters["W"].transpose(), X) + parameters["b"])).transpose()
    accuracy = (1-np.mean(np.abs(Ypredict-Y)))*100
    return accuracy

X, Y = loadIrisData("./data/Iris.csv")
Y = np.reshape(Y, (Y.shape[0], 1))
parameters = initParameter(X.shape[0], 1)

acc0 = predict(X, parameters ,Y)
print("accuracy on training set before training is {}%".format(acc0))

p, costs = trainLR(X, Y, parameters, 50, 0.001)
costs = np.squeeze(costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(0.005))

acc1 = predict(X, parameters ,Y)
print("accuracy on training set after training is {}%".format(acc1))

plt.show()