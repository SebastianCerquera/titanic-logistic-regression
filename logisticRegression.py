"""
logisticRegression.py
=====================

This module contains functions for running logistic regression on data in the
form of numpy arrays.

"""

import numpy as np


def sigmoid(x):
    """Performs the sigmoid/logistic function element-wise on an array.

    Parameters
    ----------
    x : The array to be evaluated

    Returns
    -------
    The results of the sigmoid function (same dimensions as the original array)
    """
    return 1.0 / (1.0 + np.exp(-1.0*x))


def safeLog(x):
    """Performs a log evaluation that will now blow up as x -> 0.

    Parameters
    ----------
    x : The array to be evaluated

    Returns
    -------
    The results of the log evaluation
    """
    xSafe = x + 1e-10
    return np.log(xSafe)


def hypothesis(features, thetas):
    """Calculates the hypothesis function predictions for a set of features and
    thetas (parameters).

    Parameters
    ----------
    features : The training data (must be m x n)
    thetas   : The current parameter values (must be n x 1)

    Returns
    -------
    The predicted labels based on the current parameters
    """
    return sigmoid(np.dot(features, thetas))


def cost(features, thetas, labels):
    """Calcualtes the cost of the current set of parameters based on the
    difference between the current predicted values and the actual values of the
    training data. Uses the cost function for logistic regression presented in
    Andrew Ng's Machine Learning lectures.

    Parameters
    ----------
    features : The training data (must be m x n)
               Here m is number of training examples and n the number of features
    thetas   : The current parameter values (must be n x 1)
    labels   : The actual labels of the training examples (must be m x 1)

    Returns
    J : the cost evaluation
    -------
    """
    m = len(features)
    hyp = hypothesis(features, thetas)

    firstTerm = labels * safeLog(hyp)
    secondTerm = (1.0-labels) * safeLog(1.0-hyp)

    J = (-1.0/m)*np.sum(firstTerm + secondTerm)

    return J


def gradientDescent(features, thetas, labels, alpha, minCostDif, maxIterations, cvFeatures=[], cvLabels=[]):
    """Performs gradient descent on the given data set to minimize the cost of
    theta theta parameters. If a set of cross-validation examples and a set of
    ideal labels is provided, the function will print output to the screen to
    track the accuracy of the the training.

    Parameters
    ----------
    features      : The training data (must be m x n)
                    Here m is number of training examples and n the number of features
    thetas        : The current parameter values (must be n x 1)
    labels        : The actual labels of the training examples (must be m x 1)
    alpha         : The learning rate
    minCostDif    : The cost minimum cost difference between iterations
    maxIterations : The maximum number of iterations of gradient descent

    cvFeatures    : (optional) the cross-validation data (must be l x n)
                    Here l is number of cross-validation examples
    cvLabels      : (optional) the actual labels of the cross-validation examples(must be l x 1)

    Returns
    -------
    The ideal thetas calcualted based on the data and conditions provided
    """
    m = len(features)

    costDif = np.inf
    counter = 0
    prevCost = cost(features, thetas, labels)

    while(costDif > minCostDif and counter < maxIterations):

        hyp = hypothesis(features, thetas)

        for i in range(len(thetas)):
            sumTerm = np.sum((hyp - labels) * np.reshape(features[:,i], [-1, 1]))
            thetas[i,0] = thetas[i,0] - alpha * ((1.0/m)*sumTerm)


        curCost = cost(features, thetas, labels)
        costDif = np.abs(curCost - prevCost)

        prevCost = curCost
        counter += 1

        #printing the current parameters performance on the cross-validation set every once in awhile
        if(counter%5000 == 0 and len(cvFeatures) > 0 and len(cvLabels) > 0):
            cvHyp = hypothesis(cvFeatures, thetas)
            cvHypRounded = np.round(cvHyp)
            print("Accuracy at " + str(counter) + " Iterations: " + str((cvHypRounded == cvLabels).sum()/len(cvLabels)))

    print("Complete...")
    print("Iterations: " + str(counter))
    print("Current Cost Function Evaluation: " + str(curCost))
    print("Current Cost Difference Between Last Iterations: " + str(costDif))

    return thetas
