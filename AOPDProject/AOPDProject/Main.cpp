#include "ACOImplementation.h"
#include "Ant.h"
#include <iostream>

using namespace std; 

int main() {

	srand(time(NULL));

	int startingVertex = 0;  // Zaczynamy od wierzcho�ka 0

	vector<vector<int>> edges = {
		{0, 2, 9, 10},  // Koszty przej�cia z wierzcho�ka 0 do 0, 1, 2, 3
		{1, 0, 6, 4},   // Koszty przej�cia z wierzcho�ka 1 do 0, 1, 2, 3
		{15, 7, 0, 8},  // Koszty przej�cia z wierzcho�ka 2 do 0, 1, 2, 3
		{6, 3, 12, 0}   // Koszty przej�cia z wierzcho�ka 3 do 0, 1, 2, 3
	};

	float alpha = 1.5;    // Wp�yw feromon�w
	float beta = 3.0;     // Wp�yw widoczno�ci (odwrotno�ci kosztu)
	int numberOfVertexes = 4;  // Liczba wierzcho�k�w
	int colonySize = 20;       // Liczba mr�wek w kolonii
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