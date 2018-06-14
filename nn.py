# neural network with 1 hidden layer
import numpy as np
# strange bug for mac matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv

# lets write a loader class for that
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
    return np.squeeze(np.asarray(X.transpose())), np.reshape(np.squeeze(np.asarray(np.matrix([Y]).transpose())),(len(Y),1))

# sigmoid, should be put in utils
def sigmoid(z):
    return 1/(1+np.exp(-z))

# initParameter
def initParameter(inputSize, hiddenSize, outputSize):
    params = {}
    params["W1"] = np.random.rand(inputSize, hiddenSize)
    params["b1"] = np.zeros((hiddenSize, 1), dtype=float)
    params["W2"] = np.random.rand(hiddenSize, outputSize)
    params["b2"] = np.zeros((outputSize, 1), dtype=float)
    return params

# feed_forward
def feed_forward(parameters, X):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1.transpose(), X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2.transpose(), A1) + b2
    A2 = sigmoid(Z2)
    return {
        "A1": A1.transpose(),
        "A2": A2.transpose()
    }

# back_propagate
def back_propagate(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]
    m = Y.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.transpose(), dZ2)/m
    db2 = np.sum(dZ2, axis=0, keepdims=True)/m
    dZ1 = np.dot(dZ2, parameters["W2"].transpose())*(1-np.power(A1,2))
    dW1 = np.dot(X, dZ1)/m
    db1 = (np.sum(dZ1, axis=0, keepdims=True)/m).transpose()
    return {
        "dW2": dW2,
        "db2": db2,
        "dW1": dW1,
        "db1": db1,
    }

# updateParameters
def updateParameters(parameters, grad, learningRate):
    parameters["W2"] = parameters["W2"] - learningRate * grad["dW2"]
    parameters["b2"] = parameters["b2"] - learningRate * grad["db2"]
    parameters["W1"] = parameters["W1"] - learningRate * grad["dW1"]
    parameters["b1"] = parameters["b1"] - learningRate * grad["db1"]
    return parameters

# computeLoss
def computeLoss(Yhat, Y):
    return -Y*np.log(Yhat)-(1-Y)*np.log(1-Yhat)

# mean sum of loss
def computeCost(Yhat, Y):
    loss = computeLoss(Yhat, Y)
    return np.sum(loss, axis=0, keepdims=True)/Yhat.shape[0]

# trainNN
def trainNN(X, Y, parameters, iterations, learningRate):
    costs = []
    for i in range(0, iterations):
        cache = feed_forward(parameters, X)
        grad = back_propagate(X, Y, cache, parameters)
        cost = computeCost(cache["A2"], Y)
        costs.append(cost)
        if ((i%200)==0):
            print("cost of iteration {} is {}".format(i, cost))
        parameters = updateParameters(parameters, grad, learningRate)
    return parameters, costs

# predict
def predict(X, parameters, Y):
    cache = feed_forward(parameters, X)
    Ypredict = np.round(cache["A2"]).transpose()
    accuracy = (1-np.mean(np.abs(Ypredict-Y)))*100
    return accuracy

X, Y = loadIrisData("./data/Iris.csv")
print(X.shape)
parameters = initParameter(X.shape[0], 5, Y.shape[1])
# 
acc0 = predict(X, parameters ,Y)
print("accuracy on training set before training is {}%".format(acc0))
# 
parameters, costs = trainNN(X, Y, parameters, 800, 0.002)
costs = np.squeeze(costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(0.005))
#
acc1 = predict(X, parameters ,Y)
print("accuracy on training set after training is {}%".format(acc1))
#
plt.show()