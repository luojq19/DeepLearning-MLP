""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = 0.

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
        self.loss = 1 / (2 * len(logit)) * np.sum((logit - gt)**2)
        self.acc = np.sum(np.argmax(logit, axis=1) == np.argmax(gt, axis=1)) / len(gt)
        # debug
        # print(logit[:10])
        # print(gt[:10])
        # print(np.argmax(logit, axis=1)[:10])
        # print(np.argmax(gt, axis=1)[:10])
        # print(self.accu)
        # input()
        ############################################################################

        return self.loss

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        return self.logit - self.gt

        ############################################################################
