""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

def softmax(x):
    x_exp = np.exp(x)
    x_sum = x_exp.sum(axis=1, keepdims=True)
    s = x_exp / x_sum
    return s

class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = np.zeros(1, dtype='f')

    def forward(self, logit, gt):
        """
          Inputs: (minibatch)
          - logit: forward results from the last FCLayer, shape(batch_size, 10)
          - gt: the ground truth label, shape(batch_size, 10)
        """

        ############################################################################
        # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch, and
        # store in self.accu and self.loss respectively.
        # Only return the self.loss, self.accu will be used in solver.py.
        self.logit = logit
        self.gt = gt
        self.logit_sm = softmax(logit)
        self.loss = -np.sum(gt * np.log(self.logit_sm)) / len(logit)
        self.acc = np.sum(np.argmax(self.logit_sm, axis=1) == np.argmax(gt, axis=1)) / len(gt)

        ############################################################################

        return self.loss


    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        return self.logit_sm - self.gt

        ############################################################################
