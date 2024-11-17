#include "ACOImplementation.h"
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include "kernel.h"


std::mutex pheromoneMutex;
namespace GPU {

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

    int* ACOImplementation::runAcoAlgorith(int numberOfIterations)
    {
        int startingVertexForAnt = startingVertex;
        int chosenVertex;


        //Copy eges and pheromone matrix into GPU memory
        std::vector<int> flatEdges;
        for (const auto& row : edges) {
            flatEdges.insert(flatEdges.end(), row.begin(), row.end());
        }

        int* d_edges;
        cudaMalloc(&d_edges, flatEdges.size() * sizeof(int));
        cudaMemcpy(d_edges, flatEdges.data(), flatEdges.size() * sizeof(int), cudaMemcpyHostToDevice);

        std::vector<float> flatPheromone;
        for (const auto& row : pheromoneMatrix) {
            flatEdges.insert(flatEdges.end(), row.begin(), row.end());
        }

        float* d_pheromoneMatrix;
        cudaMalloc(&d_pheromoneMatrix, flatPheromone.size() * sizeof(int));
        cudaMemcpy(d_pheromoneMatrix, flatPheromone.data(), flatPheromone.size() * sizeof(int), cudaMemcpyHostToDevice);

        for (int j = 0; j < numberOfIterations; j++) {
            for (int i = 0; i < colonySize; i++) {
                int* singleSolution = (int*)malloc(edges.size() * sizeof(int));
                while (startingVertexForAnt == startingVertex) {
                    startingVertexForAnt = rand() % edges.size();
                }
                singleSolution[0] = startingVertex;
                colony.push_back(singleSolution);
                startingVertexForAnt = startingVertex;
            }
            
            int* d_solutions;
            cudaMalloc(&d_solutions, colonySize * sizeof(int));
            cudaMemcpy(d_solutions, colony.data(), colony.size() * sizeof(int), cudaMemcpyHostToDevice);
            //do dopracowania (1 oznacza ilosc blokow, 1024 ilosc watkow na blok)
            findSolutions <<<1, colonySize>>> (d_solutions, d_pheromoneMatrix, d_edges, numberOfVertexes);
            cudaDeviceSynchronize();

            cudaMemcpy(colony.data(), d_solutions, colony.size() * sizeof(int), cudaMemcpyDeviceToHost);

            


            //evaporation
            evaporatePheromoneCAS(1, 0.1, colony);
            colony.resize(0);
        }
        return result;
    }

    void ACOImplementation::evaporatePheromoneCAS(float Qcycl, float pheromoneEvaporationRate, vector<int*> colony)
    {
        int cost;

        evaporatePheromone(pheromoneEvaporationRate);

        for (int* antSolution : colony)
        {
            cost = calculateSolutionCost(antSolution);
            if (cost < minCost)
            {
                minCost = cost;
                result = antSolution;
            }

            for (int i = 0; i < numberOfVertexes-1; i++)
            {
                pheromoneMatrix[antSolution[i]][antSolution[i + 1]] += (float)Qcycl / cost;
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

    int ACOImplementation::calculateSolutionCost(int* solution)
    {
        int cost = 0;
        for (int i = 0; i < numberOfVertexes-1; i++)
        {
            cost += edges[solution[i]][solution[i + 1]];
        }

        cost += edges[startingVertex][solution[0]];					
        cost += edges[solution[numberOfVertexes-1]][startingVertex];

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
        int* solution = new int[numberOfVertexes];

        int randIndexI, randIndexJ;

        for (int i = 1; i < numberOfVertexes; i++) solution[i] = i;

        for (int i = 0; i < numberOfVertexes; i++)
        {
            randIndexI = rand() % numberOfVertexes;	// toss index (0 , solution-1)
            randIndexJ = rand() % numberOfVertexes;
            swap(solution[randIndexI], solution[randIndexJ]);
        }

        //Divide value as there is high probability that this is not even close 
        //to the optimal value
        return calculateSolutionCost(solution) * 0.6f;
    }
}
__device__ float calculateDenominator(
    int* visitedVertexes, int visitedCount, int lastVertex, float* pheromoneMatrix, int* edgesMatrix,
    int numberOfVertexes, float alpha, float beta) {

    float denominator = 0;

    for (int i = 1; i < numberOfVertexes; i++) {
        // Check if vertex is visited
        bool isVisited = false;
        for (int j = 0; j < visitedCount; j++) {
            if (visitedVertexes[j] == i) {
                isVisited = true;
                break;
            }
        }
        if (isVisited || i == lastVertex) continue;

        float edgeCost = edgesMatrix[lastVertex * numberOfVertexes + i];
        if (edgeCost != 0) {
            denominator += powf(pheromoneMatrix[lastVertex * numberOfVertexes + i], alpha) * powf(1.0f / edgeCost, beta);
        }
        else {
            denominator += powf(pheromoneMatrix[lastVertex * numberOfVertexes + i], alpha) * powf(1.0f / 0.1f, beta);
        }
    }

    return denominator;
}

__device__ int choseVertexByProbability(
    int* visitedVertexes, int visitedCount, int lastVisitedVertex, float alpha, float beta,
    float* pheromoneMatrix, int* edgesMatrix, int numberOfVertexes) {

    float probability, toss = 0.8f, nominator, denominator, cumulativeSum = 0.0f;
    denominator = calculateDenominator(visitedVertexes, visitedCount, lastVisitedVertex, pheromoneMatrix, edgesMatrix, numberOfVertexes, alpha, beta);

    float* chances = new float[numberOfVertexes];
    int* vertices = new int[numberOfVertexes];
    int validVertexCount = 0;

    for (int i = 1; i < numberOfVertexes; i++) {
        bool isVisited = false;
        for (int j = 0; j < visitedCount; j++) {
            if (visitedVertexes[j] == i) {
                isVisited = true;
                break;
            }
        }
        if (isVisited || i == lastVisitedVertex) continue;

        float edgeCost = edgesMatrix[lastVisitedVertex * numberOfVertexes + i];
        if (edgeCost != 0) {
            nominator = powf(pheromoneMatrix[lastVisitedVertex * numberOfVertexes + i], alpha) * powf(1.0f / edgeCost, beta);
        }
        else {
            nominator = powf(pheromoneMatrix[lastVisitedVertex * numberOfVertexes + i], alpha) * powf(1.0f / 0.1f, beta);
        }
        probability = nominator / denominator;

        chances[validVertexCount] = probability;
        vertices[validVertexCount] = i;
        validVertexCount++;
    }

    // Sort by probability (naive bubble sort, CUDA doesn't support std::sort)
    for (int k = 0; k < validVertexCount - 1; k++) {
        for (int l = 0; l < validVertexCount - k - 1; l++) {
            if (chances[l] < chances[l + 1]) {
                float tempProb = chances[l];
                int tempVertex = vertices[l];
                chances[l] = chances[l + 1];
                vertices[l] = vertices[l + 1];
                chances[l + 1] = tempProb;
                vertices[l + 1] = tempVertex;
            }
        }
    }
    //BRAKUJE TUTAJ RANDOMOWEJ LICZBY TAK ZWANGEO TOSSSSAAAAAAAAAAAAA

    for (int i = 0; i < validVertexCount; i++) {
        cumulativeSum += chances[i];
        if (cumulativeSum > toss) return vertices[i];
    }

    return -1; // Fallback in case of numerical issues
}


__global__ void findSolutions(int* solutionsPointer, float* pheromoneMatrix, int* edgesMatrix, int numberOfVertexes) {
    int threadId = threadIdx.x;

    // Each thread handles one solution
    int* solution = &solutionsPointer[threadId * numberOfVertexes];

    int visitedCount = 1;

    float alpha = 1.0f, beta = 3.0f; // Example parameters
    while (visitedCount < numberOfVertexes) {
        int lastVisitedVertex = solution[visitedCount - 1];
        int nextVertex = choseVertexByProbability(solution, visitedCount, lastVisitedVertex, alpha, beta, pheromoneMatrix, edgesMatrix, numberOfVertexes);
        solution[visitedCount] = nextVertex;
        visitedCount++;
    }
}