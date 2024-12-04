﻿#include "AODPProject/gpu/ACOImplementation.h"
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstring>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <ctime>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ float calculateDenominator(
    int* visitedVertexes, int visitedCount, int lastVertex, float* probabilityResult,
    int numberOfVertexes) {

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

        denominator += probabilityResult[lastVertex*numberOfVertexes+i];
    }

    return denominator;
}

__global__ void calculateProbability(
    float* resultProbability, 
    float* pheromoneMatrix, int* edgesMatrix, int numberOfVertexes) {

    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId > numberOfVertexes * numberOfVertexes) return;
    float alpha = 1.0f, beta = 3.0f; // Example parameters

    float edgeCost = edgesMatrix[threadId];
    float nominator = 0;
    if (edgeCost != 0) {
        nominator = powf(pheromoneMatrix[threadId], alpha) * powf(1.0f / edgeCost, beta);
    }
    else {
        nominator = powf(pheromoneMatrix[threadId], alpha) * powf(1.0f / 0.1f, beta);
    }
    resultProbability[threadId] = nominator;
}

__global__ void evapouratePheromoneD(float* pheromoneMatrix, float rate,int  numberOfVertexes){

    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId > numberOfVertexes * numberOfVertexes) return;
    pheromoneMatrix[threadId] *= rate;
}
__global__ void leavePheromone(float* pheromoneMatrix, int* edgesMatrix, int* ants, int numberOfVertexes, float Qcycl) {
    
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId > numberOfVertexes * numberOfVertexes) return;
    auto cost = 0;
    for(auto i = 1; i < numberOfVertexes; i++) {
        cost += edgesMatrix[ants[threadId * numberOfVertexes +i-1] * numberOfVertexes + ants[i]];
    }
    for(auto i = 1; i < numberOfVertexes; i++) {
        pheromoneMatrix[ants[threadId * numberOfVertexes +i-1] * numberOfVertexes + ants[i]] += (float)Qcycl/cost;
    }
}

__device__ int choseVertexByProbability(
    int* visitedVertexes, int visitedCount, int lastVisitedVertex, float alpha, float beta,
    float* resultProb, int numberOfVertexes, curandState &state) {

    float probability, toss = curand_uniform_double(&state), nominator, denominator, cumulativeSum = 0.0f;
    denominator = calculateDenominator(visitedVertexes, visitedCount, lastVisitedVertex, resultProb, numberOfVertexes);

    int validVertexCount = 0;
    int notVisited = -1;

    for (int i = 0; i < numberOfVertexes; i++) {
        bool isVisited = false;
        for (int j = 0; j < visitedCount; j++) {
            if (visitedVertexes[j] == i) {
                isVisited = true;
                break;
            }
        }
        if (isVisited || i == lastVisitedVertex) continue;
        else notVisited = i;
        nominator = resultProb[lastVisitedVertex * numberOfVertexes + i];
        probability = nominator / denominator;

        cumulativeSum += probability;
        if (!(cumulativeSum == cumulativeSum) || cumulativeSum > toss) {
            return i;
        }
        validVertexCount++;
    }

    return notVisited; // Fallback in case of numerical issues
}


__global__ void findSolutions(int* solutionsPointer, float* resultProb, int numberOfVertexes) {

    //extern __shared__ int sharedInt[];
    //float* sharedFloat = (float*)(&sharedInt[blockDim.x * numberOfVertexes]);

    
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId > numberOfVertexes * numberOfVertexes) return;

    //int* vertices = &sharedInt[numberOfVertexes * threadId]; 
    //float* chances = &sharedFloat[threadId * numberOfVertexes];

    //float* chances;
    //int* vertices;
    //cudaMalloc(&chances, numberOfVertexes * sizeof(float));
    //cudaMalloc(&vertices, numberOfVertexes * sizeof(int));

    curandState state;
    curand_init((unsigned long long)clock() + threadId, 0, 0, &state);

    // Each thread handles one solution
    int* solution = &solutionsPointer[threadId * numberOfVertexes];
    int lastVisitedVertex = (int)(curand_uniform(&state) * numberOfVertexes);
    solution[0] = lastVisitedVertex;

    int visitedCount = 1;

    float alpha = 1.0f, beta = 3.0f; // Example parameters
    while (visitedCount < numberOfVertexes) {
        int nextVertex = choseVertexByProbability(solution, visitedCount, lastVisitedVertex, alpha, beta, resultProb, numberOfVertexes, state);
        solution[visitedCount] = nextVertex;
        lastVisitedVertex = nextVertex;
        visitedCount++;
    }

    //cudaFree(chances);
    //cudaFree(vertices);
}

namespace GPU {

    void ACOImplementation::init(int startingVertex, std::vector<std::vector<int>> edges, float alpha, float beta, int numberOfVertexes, int colonySize)
    {
        this->startingVertex = startingVertex;
        this->edges = edges;
        this->alpha = alpha;
        this->beta = beta;
        this->colonySize = colonySize;
        this->numberOfVertexes = numberOfVertexes;
        this->result = (int*)malloc(numberOfVertexes * sizeof(int));
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
            flatPheromone.insert(flatPheromone.end(), row.begin(), row.end());
        }

        float* d_pheromoneMatrix;
        cudaMalloc(&d_pheromoneMatrix, flatPheromone.size() * sizeof(float));
        cudaMemcpy(d_pheromoneMatrix, flatPheromone.data(), flatPheromone.size() * sizeof(float), cudaMemcpyHostToDevice);

        float* d_probMatrix;
        cudaMalloc(&d_probMatrix, numberOfVertexes * numberOfVertexes * sizeof(float));

        int* d_colony;
        cudaMalloc(&d_colony, colonySize * edges.size() * sizeof(int));

        int* h_colony = (int*)malloc(colonySize * edges.size() * sizeof(int));
        for (int j = 0; j < numberOfIterations; j++) {
            
            /*for (int i = 0; i < colonySize; i++) {
                while (startingVertexForAnt == startingVertex) {
                    startingVertexForAnt = rand() % edges.size();
                }
                h_colony[i*edges.size()] = startingVertex;
                startingVertexForAnt = startingVertex;
            }*/
            
            //cudaMemcpy(d_colony, h_colony, colonySize * sizeof(int**), cudaMemcpyHostToDevice);
            //cudaMalloc(&d_solutions, colonySize * sizeof(int));
            //cudaMemcpy(d_solutions, colony.data(), colony.size() * sizeof(int), cudaMemcpyHostToDevice);
            int blockMaxSize = -1;
            int threadsPerBlock = 32;
            //int numberOfBlocks = colonySize/threadsPerBlock + 1;

            //int sharedMemorySize = threadsPerBlock * numberOfVertexes * (sizeof(float) + sizeof(int));
            int sharedMemorySize = blockMaxSize;
            //int threadsPerBlock =  sharedMemorySize / (numberOfVertexes * (sizeof(float) + sizeof(int)));
            int numberOfBlocks = colonySize/threadsPerBlock + 1;


            //int* antsData;
            //cudaMalloc(&antsData, numberOfVertexes * colonySize * (sizeof(float) + sizeof(int)));

            //sharedMemorySize += sizeof(int) - sharedMemorySize % sizeof(int);
            //printf("Colony size is: %d\nNumber of vertices: %d\nBlock max shared mem size: %d\nStarting Kernel on %d blocks each %d threads with %d bytes of shared memory\n", colonySize, numberOfVertexes, blockMaxSize, numberOfBlocks, threadsPerBlock, sharedMemorySize);
            //do dopracowania (1 oznacza ilosc blokow, 1024 ilosc watkow na blok)
            calculateProbability<<<numberOfVertexes * numberOfVertexes / threadsPerBlock, threadsPerBlock>>> (d_probMatrix, d_pheromoneMatrix, d_edges, numberOfVertexes );
            findSolutions <<<numberOfBlocks, threadsPerBlock>>> (d_colony, d_probMatrix, numberOfVertexes );
            evapouratePheromoneD<<<numberOfVertexes * numberOfVertexes / threadsPerBlock, threadsPerBlock>>>(d_pheromoneMatrix, 0.1, numberOfVertexes);
            leavePheromone <<<numberOfBlocks, threadsPerBlock>>> (d_pheromoneMatrix,  d_edges, d_colony, numberOfVertexes, 1.0); 

            //cudaFree(antsData);
            //cudaMemcpy(colony.data(), d_solutions, colony.size() * sizeof(int), cudaMemcpyDeviceToHost);
            //TUTAJ są kopiuowane tylko wskaźniki do tablic z rozwiazaniami mrówek a nie same ścieżki z mrówkami

            


            //evaporation
        }
        cudaMemcpy(h_colony, d_colony, edges.size() * colonySize * sizeof(int), cudaMemcpyDeviceToHost);
        evaporatePheromoneCAS(1, 0.1, h_colony);
        return result;
    }

    bool containsOnlyCities(int* path, int numberOfCities) {
        auto zero_corrected = false;
        for(auto i = 0; i < numberOfCities; i++) {
            if(path[i] > numberOfCities || path[i] < 0) {
                if (zero_corrected) return false;
                zero_corrected = true;
                path[i] = 0;
            }
        }
        return true;
    }

    void ACOImplementation::evaporatePheromoneCAS(float Qcycl, float pheromoneEvaporationRate, int* colony)
    {
        int cost;

        evaporatePheromone(pheromoneEvaporationRate);
        int* antSolution;
        int* _result = nullptr;
        for (auto ant = 0; ant < colonySize; ant++)
        {
            antSolution = &colony[ant * edges.size()];
            //if(containsOnlyCities(antSolution, numberOfVertexes)) continue;
            cost = calculateSolutionCost(antSolution);
            if (cost < minCost)
            {
                minCost = cost;
                _result = antSolution;
            }

            for (int i = 0; i < numberOfVertexes-1; i++)
            {
                pheromoneMatrix[antSolution[i]][antSolution[i + 1]] += (float)Qcycl / cost;
            }
        }
        if (_result != nullptr) {
            std::memcpy(result, _result, numberOfVertexes * sizeof(int));
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
        std::vector<float> tempVec;

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

        for (int i = 0; i < numberOfVertexes; i++) solution[i] = i;

        for (int i = 0; i < numberOfVertexes; i++)
        {
            randIndexI = rand() % numberOfVertexes;	// toss index (0 , solution-1)
            randIndexJ = rand() % numberOfVertexes;
            std::swap(solution[randIndexI], solution[randIndexJ]);
        }

        //Divide value as there is high probability that this is not even close 
        //to the optimal value
        return calculateSolutionCost(solution) * 0.6f;
    }
}

