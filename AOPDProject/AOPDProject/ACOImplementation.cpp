#include "ACOImplementation.h"
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
std::mutex pheromoneMutex;

float ACOImplementation::calculateDenominator(Ant ant, int lastVertex, float alpha, float beta)
{
	float denominator = 0;

	for (int i = 1; i < numberOfVertexes; i++)
	{
		if (find(ant.visitedVertexes.begin(), ant.visitedVertexes.end(), i) != ant.visitedVertexes.end()) continue; 
		if (i == lastVertex) continue;

		// cant divide by zero, if cost of edge is zero take a small value instead
		if (edges[lastVertex][i] != 0) { 
			denominator += pow(pheromoneMatrix[lastVertex][i], alpha) * pow(((float)1 / edges[lastVertex][i]), beta); }
		else {
			denominator += pow(pheromoneMatrix[lastVertex][i], alpha) * pow(((float)1 / 0.1), beta);
		}
	}

	return denominator;
}

int ACOImplementation::choseVertexByProbability(Ant ant, float alpha, float beta)
{
	float probability, toss, nominator, denominator, cumulativeSum = 0.0;		
	int lastVisitedVertex = ant.visitedVertexes.back();

	vector<VertexProbability> chances;

	denominator = calculateDenominator(ant, lastVisitedVertex, alpha, beta);

	for (int i = 1; i < numberOfVertexes; i++)
	{
		if (find(ant.visitedVertexes.begin(), ant.visitedVertexes.end(), i) != ant.visitedVertexes.end()) continue; 
		if (i == lastVisitedVertex) continue;

		// if not visited, calculate probability
		// (Tij)^(alpha) * (Nij)^Beta, where Tij -> pheromoneMatrix[i][j] and Nij -> 1/Lij [cryterium visibility] -> Lij = length, cost so adjencyMatrix
		// cant divide by zero, if cost of edge is zero take a small value instead
		if (edges[lastVisitedVertex][i] != 0) {
			nominator = (float)pow(pheromoneMatrix[lastVisitedVertex][i], alpha) * pow((float)1 / edges[lastVisitedVertex][i], beta);
		} 
		else {
			nominator = (float)pow(pheromoneMatrix[lastVisitedVertex][i], alpha) * pow((float)1 / 0.1, beta);
		}
		probability = nominator / denominator;

		VertexProbability newVertex;
		newVertex.probability = probability;
		newVertex.vertex = i;
		chances.push_back(newVertex);
	}

	sort(chances.begin(), chances.end(), VertexProbabilityGreater());

	toss = rand() % 100 / 100.0;

	for (int i = 0; i < chances.size(); i++)	
	{
		cumulativeSum += chances[i].probability;
		if (cumulativeSum > toss) return chances[i].vertex;
	}
}

void ACOImplementation::init(int startingVertex, vector<vector<int>> edges, float alpha, float beta, int numberOfVertexes, int colonySize)
{
	this->startingVertex = startingVertex;
	this->edges = edges;
	this->alpha = alpha;
	this->beta = beta;
	this->colonySize = colonySize;
	this->numberOfVertexes = numberOfVertexes;
	initializePheromoneMatrix(calculateApproximatedSolutionCost());
}

vector<int> ACOImplementation::runAcoAlgorith(int numberOfIterations)
{
	int startingVertexForAnt = startingVertex;
	int chosenVertex;
	for (int j = 0; j < numberOfIterations; j++) {
		for (int i = 0; i < colonySize; i++) {
			while (startingVertexForAnt == startingVertex) {
				startingVertexForAnt = rand() % edges.size();
			}
			Ant newAnt;
			newAnt.addNewVertex(startingVertexForAnt);
			colony.push_back(newAnt);
			startingVertexForAnt = startingVertex;
		}

		const int threadsAmount = std::thread::hardware_concurrency();

		int antsAmountPerThread = colonySize / threadsAmount;
		
		vector<thread> threads(threadsAmount);

		for (int i = 0; i < threadsAmount; i++)
		{
			int rangeBegin = i * antsAmountPerThread;
			int rangeEnd = (i != threadsAmount - 1)? rangeBegin + antsAmountPerThread : colonySize;

			threads[i] = thread([rangeBegin, rangeEnd, this] {
				for (int j = rangeBegin; j < rangeEnd; j++) {
					for (int i = 2; i < edges.size(); i++) {
						colony[j].addNewVertex(choseVertexByProbability(colony[j], alpha, beta));
					}
				}
			});
		}

		for (int i = 0; i < threadsAmount; i++)
		{
			if (threads[i].joinable()) {
				threads[i].join();
			}
		}

		//evaporation
		evaporatePheromoneDAS(1, 0.1, colony);
		colony.resize(0);
	}
	return result;
}

void ACOImplementation::evaporatePheromoneDAS(float Qdens, float pheromoneEvaporationRate, vector<Ant> colony)
{
	int cost;

	evaporatePheromone(pheromoneEvaporationRate);

	for (Ant ant : colony)
	{
		cost = calculateSolutionCost(ant.visitedVertexes);
		if (cost < minCost)
		{
			minCost = cost;
			result = ant.visitedVertexes;
		}

		for (int i = 0; i < ant.visitedVertexes.size() - 1; i++)
		{
			pheromoneMatrix[ant.visitedVertexes[i]][ant.visitedVertexes[i + 1]] += Qdens;
		}
	}
}

void ACOImplementation::evaporatePheromoneQAS(float Qquan, float pheromoneEvaporationRate, vector<Ant> colony)
{
	int cost;

	evaporatePheromone(pheromoneEvaporationRate);

	for (Ant ant : colony)
	{
		cost = calculateSolutionCost(ant.visitedVertexes);
		if (cost < minCost)
		{
			minCost = cost;
			result = ant.visitedVertexes;
		}

		for (int i = 0; i < ant.visitedVertexes.size() - 1; i++)
		{
			pheromoneMatrix[ant.visitedVertexes[i]][ant.visitedVertexes[i + 1]] += 
				(float)Qquan / edges[ant.visitedVertexes[i]][ant.visitedVertexes[i + 1]];
		}
	}
}

void ACOImplementation::evaporatePheromoneCAS(float Qcycl, float pheromoneEvaporationRate, vector<Ant> colony)
{
	int cost;

	evaporatePheromone(pheromoneEvaporationRate);

	for (Ant ant : colony)
	{
		cost = calculateSolutionCost(ant.visitedVertexes);
		if (cost < minCost)
		{
			minCost = cost;
			result = ant.visitedVertexes;
		}

		for (int i = 0; i < ant.visitedVertexes.size() - 1; i++)
		{
			pheromoneMatrix[ant.visitedVertexes[i]][ant.visitedVertexes[i + 1]] += (float)Qcycl / cost;
		}
	}
}

void ACOImplementation::evaporatePheromone(float pheromoneEvaporationRate)
{
	for (int i = 0; i < numberOfVertexes; i++)
	{
		for (int j = 0; j < numberOfVertexes; j++)
		{
			pheromoneMatrix[i][j] *= pheromoneEvaporationRate;
		}
	}
}

int ACOImplementation::calculateSolutionCost(vector<int> solution)
{
	int cost = 0;
	for (int i = 0; i < solution.size() - 1; i++)
	{
		cost += edges[solution[i]][solution[i + 1]];
	}

	cost += edges[startingVertex][solution[0]];					
	cost += edges[solution[solution.size() - 1]][startingVertex];	

	return cost;
}

void ACOImplementation::initializePheromoneMatrix(int aproximatedSolutionCost)
{
	float tau_zero = (float)colonySize / (float)aproximatedSolutionCost;
	vector<float> tempVec;

	for (int i = 0; i < numberOfVertexes; i++)
	{
		tempVec.push_back(tau_zero);
	}

	for (int i = 0; i < numberOfVertexes; i++)
	{
		pheromoneMatrix.push_back(tempVec);
	}
}

float ACOImplementation::calculateApproximatedSolutionCost()
{
	vector<int> solution;

	int randIndexI, randIndexJ;

	for (int i = 1; i < numberOfVertexes; i++) solution.push_back(i);

	for (int i = 0; i < numberOfVertexes; i++)
	{
		randIndexI = rand() % solution.size();	// toss index (0 , solution-1)
		randIndexJ = rand() % solution.size();
		swap(solution[randIndexI], solution[randIndexJ]);
	}

	//Divide value as there is high probability that this is not even close 
	//to the optimal value
	return calculateSolutionCost(solution) * 0.6f;
}
