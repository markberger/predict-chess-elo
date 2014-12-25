import numpy as np
import theano
import theano.tensor as T

from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from BaseClassifier import BaseClassifier


class TheanoFNN:

    def __init__(self, n_in, n_out, n_hidden=500, C=.0001):
        # Regularization constant
        self.C = C

        # Weight matrix that maps input to hidden layer
        self.W = theano.shared(
            value=np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    # +1 for the bias
                    size=(n_hidden + 1, n_in),
                ),
                dtype=theano.config.floatX,
            ),
            name='W',
            borrow=True,
        )

        # Weight matrix that maps hidden layer to output
        self.U = theano.shared(
            value=np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    # +1 for the bias
                    size=(n_out, n_hidden + 1),
                ),
                dtype=theano.config.floatX,
            ),
            name='U',
            borrow=True,
        )

        self._compile()

    def _compile(self):
   		X = T.matrix('X', dtype=theano.config.floatX)
   		Y = T.matrix('Y', dtype=theano.config.floatX)
   		lr = T.scalar('lr', dtype=theano.config.floatX)

		# Predict
		out = self._sym_forward(X)
		self.predict = theano.function([X], out)

		# Train
		cost, updates = self._sym_train(X, Y, lr)
		self.train = theano.function([X, Y, lr], cost, updates=updates)

		# Objective function
		objective = self._sym_objective(X, Y)
		self.objective = theano.function([X, Y], objective)

    def _sym_objective(self, X, Y):
    	return T.mean(T.abs_(X - Y))

    def _sym_forward(self, x):
        z = T.dot(self.W, x.T)
        a = T.nnet.sigmoid(z)
        out = T.dot(self.U, a)
        return out

    def _sym_cost(self, X, Y):
        R = self._sym_forward(X)
        out = T.mean(T.abs_(R.T - Y))
        return out

    def _sym_train(self, X, Y, lr):
		cost = (
			self._sym_cost(X, Y) +
			(self.C / (2 * X.shape[0])) *
			((self.W ** 2).sum() +
			(self.U ** 2).sum())
		)

		dW = theano.gradient.grad(cost, self.W)
		dU = theano.gradient.grad(cost, self.U)

		updates = [
			(self.W, self.W - lr * dW),
			(self.U, self.U - lr * dU),
		]

		return cost, updates

class FNN(BaseClassifier):

	def __init__(self):
		self._net = None

	def predict(self, X):
		predicted =	self._net.predict(X)
		results = []
		for i in xrange(predicted.shape[0]):
			results.append({'WhiteElo': predicted[i]})
		return results

	def train(self, X, Y, max_epochs=1000, batch_size=256):
		if self._net is None:
			print 'Compiling net...'
			self._net = TheanoFNN(X.shape[1], Y.shape[1])

		for i in xrange(max_epochs):
			print 'Epoch', i
			if i < 2:
				learn_rate = .1
			elif i < 90:
				learn_rate = .01
			elif i < 100:
				learn_rate = .005
			else:
				learn_rate = .001

			for j in xrange(X.shape[0] / batch_size):
				x = X[batch_size * j : (j+1) * batch_size, :]
				y = Y[batch_size * j : (j+1) * batch_size, :]
				self._net.train(x, y, learn_rate)

			print 'Calculating objective...'
			print self._net.objective(self._net.predict(X).T, Y)

			self._shuffle(X, Y)

	def transform(self, games):
		extractor = FeatureExtractor()
		vectorizer = DictVectorizer(sparse=False)
		X = vectorizer.fit_transform(extractor.transform(games))
		return X

	def _shuffle(self, X, Y):
		assert len(X) == len(Y)
		p = np.random.permutation(len(X))
		return X[p], Y[p]


class FeatureExtractor(TransformerMixin):

	def transform(self, games, **transform_params):
		featuresList = []
		for game in games:
			features = {}
			for i in xrange(len(game['Moves'])):
				label = 'Move' + str(i)
				features[label] = game['Moves'][i]

			features['NumMoves'] = len(game['Moves'])
			features['Result'] = game['Result']
			featuresList.append(features)

		return featuresList


	def fit(self, X, y=None, **fit_params):
		return self

