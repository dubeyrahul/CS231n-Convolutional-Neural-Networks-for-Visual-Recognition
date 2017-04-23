import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        # For correct class, go over all incorrect classes and add -X[i] for all incorrect classes
        # that have margin > 0
        # Reference: http://cs231n.github.io/optimization-1/
        for k in xrange(num_classes):
          if k != j:
            temp_margin = scores[k] - correct_class_score + 1
            if temp_margin > 0:
              dW[:,j] += -X[i,]
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # For incorrect class, add X[i] to the gradient
        dW[:,j] += X[i,]
  dW /= num_train
  dW += 2*reg*W

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################



  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  # print "Incoming X shape:", X.shape
  # print "W shape: ",W.shape
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(scores.shape[0]), y] # Using boolean masking
  # print "scores shape: ", scores.shape
  # print "correct class scores shape: ", correct_class_scores.shape
  # print "correct class matrix shape: ", np.matrix(correct_class_scores).shape
  # By using np.matrix, (500,) becomes (1, 500) and then taking transporse, it becomes (500, 1)
  # Then broadcasting is applied which makes it (500, 10)
  margins = np.maximum(0, scores - np.matrix(correct_class_scores).T + 1) 
  margins[np.arange(scores.shape[0]), y] = 0
  loss = np.mean(np.sum(margins, axis=1))
  loss += reg*np.linalg.norm(W, ord='fro')
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  counts = margins
  # Making all margins > 0 as 1, because these will be counted, according to the formula
  counts[margins > 0] = 1 
  # Now calculate sum across each row, that is for each example
  sum_counts = np.sum(counts, axis=1)
  # Now for correct y[i], subtract the count as per the formula for correct class
  counts[np.arange(num_train), y] -= sum_counts.T
  # Final dot product to produce the dW matrix (this is indicator function dotted with X[i], in vectorized form)
  dW = np.dot(X.T, counts)/X.shape[0]
  # Adding regularizer
  dW += 0.5*reg*W

  return loss, dW
