""" Optimizer Class """

import numpy as np

class SGD():
	def __init__(self, learningRate, weightDecay):
		self.learningRate = learningRate
		self.weightDecay = weightDecay

	# One backpropagation step, update weights layer by layer
	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:

				############################################################################
				# TODO: Put your code here
				# Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
				# Do not forget the weightDecay term.
				diff_W = self.learningRate * (layer.grad_W + self.weightDecay * layer.W)
				diff_b = self.learningRate * layer.grad_b.sum(axis=0, keepdims=True) / len(layer.grad_b)
				# debug
				# print(diff_b.shape)
				# print(layer.b.shape)
				# debug
				############################################################################

				# Weight update
				layer.W -= diff_W
				layer.b -= diff_b
