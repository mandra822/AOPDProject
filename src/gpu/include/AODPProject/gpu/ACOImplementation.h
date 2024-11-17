#pragma once
#include "Ant.h"
#include "VertexProbability.h"
#include <algorithm>
#include <stdio.h>

namespace GPU {
    class ACOImplementation {
    private:
        vector<vector<int> > edges;
        vector<vector<float> > pheromoneMatrix;
        vector<int*> colony;
        int* result;
        vector<int> tempResult;
        int colonySize;
        int minCost = INT_MAX;
        int startingVertex;
        float alpha;
        float beta;
        int numberOfVertexes;
        void evaporatePheromone(float pheromoneEvaporationRate);
        void evaporatePheromoneCAS(float Qcycl, float pheromoneEvaporationRate, vector<int*> colony);
        void initializePheromoneMatrix(int aproximatedSolutionCost);
        float calculateApproximatedSolutionCost();
    public: 
        void init(int startingVertex, vector<vector<int> > edges, float alpha, float beta, int numberOfVertexes, int colonySize);
        int* runAcoAlgorith(int numberOfIterations);
        int calculateSolutionCost(int* solution);
    };
}
