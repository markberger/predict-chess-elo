import numpy as np

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression as LR

from BaseClassifier import BaseClassifier
from MeanBaseline import MeanBaseline

MIN_NUM_MOVES = 10

class LinearRegression(BaseClassifier):

	def train(self, data, labels):
		self._baseline = MeanBaseline()
		self._baseline.train(data)

		clf1 = LR()
		self._white_linear = Pipeline([
			('feature_extractor', FeatureExtractor()),
			('vect', DictVectorizer()),
			('lr', clf1),
		])

		clf2 = LR()
		self._black_linear = Pipeline([
			('feature_extractor', FeatureExtractor()),
			('vect', DictVectorizer()),
			('lr', clf2),
		])

		train_data = [game for game in data if len(game['Moves']) >= 4]
		white_elos = [game['WhiteElo'] for game in train_data]
		black_elos = [game['BlackElo'] for game in train_data]

		print 'Training LR for white players...'
		self._white_linear.fit(train_data, white_elos)
		print 'Training LR for black players...'
		self._black_linear.fit(train_data, black_elos)

	def predict(self, data):
		results = []
		for game in data:
			if len(game['Moves']) < MIN_NUM_MOVES:
				results += self._baseline.predict([game])
				continue

			result = {}
			result['WhiteElo'] = self._white_linear.predict([game])[0]
			result['BlackElo'] = self._black_linear.predict([game])[0]
			results.append(result)

		return results

	def transform(self, data):
		return data


def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

class FeatureExtractor(TransformerMixin):

	def transform(self, games, **transform_params):
		featuresList = []
		for game in games:
			features = {}

			scores = [int(s) if s != 'NA' else 0 for s in game['Stockfish_Scores']]
			whiteScores = scores[0::2]
			blackScores = scores[1::2]

			features['WhiteDeltaMedian'] = np.median(scores)
			features['WhiteDeltaStd'] = np.std(scores)
			features['MaxWhiteDelta'] = max(scores)
			features['MinWhiteDelta'] = min(scores)
			features['WhiteScoresMedian'] = np.median(whiteScores)
			features['BlackScoresMedian'] = np.median(blackScores)
			features['MaxBlackDelta'] = max(blackScores)
			features['MinBlackDelta'] = min(blackScores)
			features['Result'] = game['Result']
			features['NumMoves'] = len(game['Moves'])

			if len(scores) >= 4:
				features['FirstMoveDelta'] = scores[0] - scores[1]
				features['SecondMoveDelta'] = scores[2] - scores[3]

			# Was min(games, 6)
			"""
			for i in xrange(len(game['Moves'])):
				label = 'Move' + str(i)
				features[label] = game['Moves'][i]

			for i in xrange(min(len(scores), 4, 2)):
				label = 'Delta' + str(i)
				features[label] = scores[i] - scores[i+1]
			"""

			for i in xrange(min(len(game['Moves']), 8)):
				features['Move' + str(i)] = game['Moves'][i]

			if(len(game['Moves']) > MIN_NUM_MOVES):
				NUM_CHUNKS = 4
				generalScores = [c for c in chunkIt(scores, NUM_CHUNKS)]
				dividedWhiteScores = [c for c in chunkIt(whiteScores, NUM_CHUNKS)]
				dividedBlackScores = [c for c in chunkIt(blackScores, NUM_CHUNKS)]

				for i in xrange(NUM_CHUNKS):
					features['OverallScoreMean' + str(i)] = np.mean(generalScores[i])
					features['WhiteMean' + str(i)] = np.mean(dividedWhiteScores[i])
					features['BlackMean' + str(i)] = np.mean(dividedBlackScores[i])

					features['OverallScoreMedian' + str(i)] = np.median(generalScores[i])
					features['WhiteMedian' + str(i)] = np.median(dividedWhiteScores[i])
					features['BlackMedian' + str(i)] = np.median(dividedBlackScores[i])

					features['OverallScoreStd' + str(i)] = np.std(generalScores[i])
					features['WhiteStd' + str(i)] = np.std(dividedWhiteScores[i])
					features['BlackStd' + str(i)] = np.std(dividedBlackScores[i])

			featuresList.append(features)

		return featuresList

	def fit(self, X, y=None, **fit_params):
		return self
