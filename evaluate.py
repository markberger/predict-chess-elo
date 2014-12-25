import numpy as np
import sklearn.cross_validation

from read_data import read_games
from classifiers import MeanBaseline
from classifiers import LinearRegression
from classifiers import FNN

NUM_FOLDS = 5

def cross_validate(clf):
	training_games, _ = read_games()
	kf = sklearn.cross_validation.KFold(len(training_games), NUM_FOLDS)
	error = 0.0
	for train_indices, test_indices in kf:
		train, test = get_splits(training_games, train_indices, test_indices)
		clf.train(train)
		predicted = clf.predict(test)
		split_error = evaluate_results(test, predicted)
		print 'Split error:', split_error
		error += split_error

	average_error = error / NUM_FOLDS
	print 'Average error over', NUM_FOLDS, 'splits:', average_error

def get_splits(games, train_indices, test_indices):
	train_split = []
	test_split = []
	for i in train_indices:
		train_split.append(games[i])
	for i in test_indices:
		test_split.append(games[i])
	return train_split, test_split

def evaluate_results(games, predicted):
	abs_error = 0.0
	for i in xrange(len(games)):
		abs_error += abs(games[i]['WhiteElo'] - predicted[i]['WhiteElo'])
		abs_error += abs(games[i]['BlackElo'] - predicted[i]['BlackElo'])

	return abs_error / (len(games)*2)

def convert_to_matrix(games):
	Y = np.zeros((len(games), 1))
	for i in xrange(len(games)):
		Y[i] = games[i]['WhiteElo']
	return Y

def evaluate_nn():
	training_games, _ = read_games()
	nn = FNN()
	i = int(round(len(training_games) * .8))
	X = nn.transform(training_games)
	training_games, testing_games = training_games[:i], training_games[i:]
	X, X_test = X[:i, :], X[i:, :]
	Y, Y_test = convert_to_matrix(training_games), convert_to_matrix(testing_games)
	nn.train(X, Y)

if __name__ == '__main__':
	#clf = LinearRegression()
	#cross_validate(clf)
	evaluate_nn()
