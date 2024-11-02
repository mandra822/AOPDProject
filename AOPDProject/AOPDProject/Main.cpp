#include "ACOImplementation.h"
#include "Ant.h"
#include <iostream>

using namespace std; 

int main() {

	srand(time(NULL));

	int startingVertex = 0;  // Zaczynamy od wierzcho³ka 0

	vector<vector<int>> edges = {
		{0, 2, 9, 10},  // Koszty przejœcia z wierzcho³ka 0 do 0, 1, 2, 3
		{1, 0, 6, 4},   // Koszty przejœcia z wierzcho³ka 1 do 0, 1, 2, 3
		{15, 7, 0, 8},  // Koszty przejœcia z wierzcho³ka 2 do 0, 1, 2, 3
		{6, 3, 12, 0}   // Koszty przejœcia z wierzcho³ka 3 do 0, 1, 2, 3
	};

	float alpha = 1.5;    // Wp³yw feromonów
	float beta = 3.0;     // Wp³yw widocznoœci (odwrotnoœci kosztu)
	int numberOfVertexes = 4;  // Liczba wierzcho³ków
	int colonySize = 20;       // Liczba mrówek w kolonii
	ACOImplementation aco;
	aco.init(startingVertex, edges, alpha, beta, numberOfVertexes, colonySize);

	vector<int> result = aco.runAcoAlgorith(100);
	cout << aco.calculateSolutionCost(result) << endl;

	for (int i = 0; i < result.size(); i++)
	{
		cout << result[i] << " ";
	}

	return 0;
}