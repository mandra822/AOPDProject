#pragma once
#include "Ant.h"
#include "VertexProbability.h"
#include <algorithm>
#include <climits>

namespace NONCONCURRENT {
    class ACOImplementation {
    private:
        vector<vector<int>> edges;
        vector<vector<float>> pheromoneMatrix;
        vector<Ant> colony;
        vector<int> result;
        vector<int> tempResult;
        int colonySize;
        int minCost = INT_MAX;
        int startingVertex;
        float alpha;
        float beta;
        int numberOfVertexes;
        float calculateDenominator(Ant ant, int lastVertex, float alpha, float beta);
        void updatePheromone();
        int choseVertexByProbability(Ant ant, float alpha, float beta);
        void evaporatePheromoneQAS(float Qquan, float pheromoneEvaporationRate, vector<Ant> colony);
        void evaporatePheromone(float pheromoneEvaporationRate);
        void evaporatePheromoneCAS(float Qcycl, float pheromoneEvaporationRate, vector<Ant> colony);
        
        void initializePheromoneMatrix(int aproximatedSolutionCost);
        float calculateApproximatedSolutionCost();
    public: 
        void init(int startingVertex, vector<vector<int>> edges, float alpha, float beta, int numberOfVertexes, int colonySize);
        vector<int> runAcoAlgorith(int numberOfIterations);
        void evaporatePheromoneDAS(float Qdens, float pheromoneEvaporationRate, vector<Ant> colony);
        int calculateSolutionCost(vector<int> solution);
    };
}
