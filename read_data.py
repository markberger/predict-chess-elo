UCI_DATA_PATH = 'data/data_uci.pgn'
STOCKFISH_DATA_PATH = 'data/stockfish.csv'

def read_games():
	with open(UCI_DATA_PATH) as gamesFile:
		with open(STOCKFISH_DATA_PATH) as movesFile:
			trainingGames, testingGames = parseGames(gamesFile, movesFile)

	return trainingGames, testingGames

def parseGames(gamesFile, movesFile):
	# Discard csv header
	movesFile.readline()

	trainingGames = []
	testingGames = []
	game = {}

	for line in gamesFile:
		line = line.strip()

		if not line:
			continue

		if line.startswith('['):
			line = line.split('"')
			key = line[0][1:].strip()
			value = line[1]

			if line[0].startswith('[WhiteElo') or line[0].startswith('[BlackElo'):
				game[key] = int(value)
			else:
				game[key] = value

		else:
			game['Moves'] = line.split()[:-1]
			game['Stockfish_Scores'] = movesFile.readline().split(',')[1].split()
			if 'WhiteElo' in game:
				trainingGames.append(game)
			else:
				testingGames.append(game)
			game = {}

	return trainingGames, testingGames

if __name__ == '__main__':
	trainingGames, testingGames = read_games()
	print 'Number of training games:', len(trainingGames)
	print 'Number of testing games:', len(testingGames)
	numWith10Plus = 0
	for game in trainingGames:
		if len(game['Moves']) >= 13:
			numWith10Plus += 1
	print numWith10Plus
