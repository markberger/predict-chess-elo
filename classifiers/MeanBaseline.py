import numpy as np

from BaseClassifier import BaseClassifier

class MeanBaseline(BaseClassifier):

	_median_elo = None

	def train(self, data):
		white_elos = [d['WhiteElo'] for d in data]
		black_elos = [d['BlackElo'] for d in data]

		self._median_elo = np.median(np.array(white_elos + black_elos))

	def predict(self, data):
		result = {
			'WhiteElo': self._median_elo,
			'BlackElo': self._median_elo,
		}

		return [result for i in xrange(len(data))]
