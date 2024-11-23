#pragma once
#include "AODPProject/gpu/VertexProbability.h"
#include <algorithm>
#include <stdio.h>

namespace GPU {
    class ACOImplementation {
    private:
        std::vector<std::vector<int> > edges;
        std::vector<std::vector<float> > pheromoneMatrix;
        std::vector<int*> colony;
        int* result;
        std::vector<int> tempResult;
        int colonySize;
        int minCost = INT_MAX;
        int startingVertex;
        float alpha;
        float beta;
        int numberOfVertexes;
        void evaporatePheromone(float pheromoneEvaporationRate);
        void evaporatePheromoneCAS(float Qcycl, float pheromoneEvaporationRate, int* colony);
        void initializePheromoneMatrix(int aproximatedSolutionCost);
        float calculateApproximatedSolutionCost();
    public: 
        void init(int startingVertex, std::vector<std::vector<int> > edges, float alpha, float beta, int numberOfVertexes, int colonySize);
        int* runAcoAlgorith(int numberOfIterations);
        int calculateSolutionCost(int* solution);
    };
}
