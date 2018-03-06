import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
  
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    ##############################################################################
    # The loss function is:                                                      #
    # L(W) = - sum_{i = 1}^{N} sum_{k = 1}^{C} 1{y[i] == k} * log                #
    # (exp(x[i].dot(W[:, k])) / (sum_{j = 1}^{C} exp(x[i].dot(W[:, j])))         #
    ##############################################################################
    # The gradient of the loss function wrt W[:, k] is:                          #
    #                                                                            #
    # grad_{W[: , k]} = - sum_{i = 1}^{N} [x[i] * (1{y[i] == k} -                #
    #                                            Pr(y[i] = k | x[i]; W))]        #
    ##############################################################################
    for i in range(num_train):
        class_scores = X[i].dot(W)
        # For numeric stability, we subtract the maximum from
        # each of the class_scores. Since we take exponentials
        # and then divide by the sum, we get back the same result.
        class_scores -= np.max(class_scores)
        class_scores_exp = np.exp(class_scores).reshape((10, 1))

        class_probs = class_scores_exp / np.sum(class_scores_exp)
        prob_i = class_probs[y[i]]  # the prob for the correct class
        loss += (-1) * np.log(prob_i)

        d = np.zeros((num_classes, 1))
        d[y[i]] = 1
        d -= class_probs
        x = X[i].reshape((1, X[i].shape[0]))
        dW += (-1) * d.dot(x).T

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    dW /= num_train
    loss /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)  # scores = N * C
    # For numerical stability
    scores -= np.max(scores, axis=1).reshape((num_train, 1))
    scores_exp = np.exp(scores)  # scores_exp = N * C

    sum_scores = np.sum(scores_exp, axis=1, keepdims=True)
    prob = scores_exp / sum_scores  # prob = N * C

    prob_i = prob[np.arange(num_train), y]

    d = np.zeros_like(prob)
    d[np.arange(num_train), y] = 1
    d -= prob

    dW += (-1) * X.T.dot(d)
    loss += (-1) * np.sum(np.log(prob_i))

    dW /= num_train
    loss /= num_train
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    return loss, dW
