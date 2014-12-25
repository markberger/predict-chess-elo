import csv

from read_data import read_games
from classifiers import LinearRegression

SUBMISSION_NAME = 'submission.csv'

def create_submission():
	training_games, test_games = read_games()
	clf = LinearRegression()
	clf.train(training_games)

	with open(SUBMISSION_NAME, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['Event', 'WhiteElo', 'BlackElo'])
		predicted = clf.predict(test_games)
		for i in xrange(len(test_games)):
			row = [
				test_games[i]['Event'],
				predicted[i]['WhiteElo'],
				predicted[i]['BlackElo'],
			]
			writer.writerow(row)

if __name__ == '__main__':
	create_submission()