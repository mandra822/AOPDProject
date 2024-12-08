#include "AODPProject/gpu/ACOImplementation.h"
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdio.h>
#include <iostream>

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
    float* chances, int visitedCount,
    int numberOfVertexes) {

    float denominator = 0.0f;

    for (int i = visitedCount; i < numberOfVertexes - 1; i++) {
        denominator += chances[i];
    }

    return denominator;
}

__global__ void evapouratePheromoneD(float* pheromoneMatrix, float rate,int  numberOfVertexes){

    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId > numberOfVertexes * numberOfVertexes) return;
    pheromoneMatrix[threadId] *= rate;
}

__global__ void leavePheromone(float* pheromoneMatrix, int* edgesMatrix, int* ants, int numberOfVertexes, float Qcycl, int* costs, int colonySize) {
    
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId > colonySize) return;
    auto cost = 0;
    for(auto i = 1; i < numberOfVertexes; i++) {
        cost += edgesMatrix[ants[threadId * numberOfVertexes +i-1] * numberOfVertexes + ants[threadId * numberOfVertexes +i]];
    }
    costs[threadId] = cost;
    for(auto i = 1; i < numberOfVertexes; i++) {
        pheromoneMatrix[ants[threadId * numberOfVertexes + (i - 1)] * numberOfVertexes + ants[threadId * numberOfVertexes + i]] += (float)Qcycl / (float)cost;
    }
}

__device__ int choseVertexByProbability(
    int* sharedInt, float* chances, int visitedCount, int numberOfVertexes, curandState &state) {

    float toss = curand_uniform_double(&state), cumulativeSum = 0.0f;
    for (int i = visitedCount; i < numberOfVertexes; i++) {

        cumulativeSum += chances[i];
        if (!(cumulativeSum == cumulativeSum) || cumulativeSum > toss) {
            return sharedInt[i];
        }
    }
    return sharedInt[numberOfVertexes - 1];
}

__device__ void calculateNominatorToShared(
    int* vertexes,
    float* chances, 
    int ownVertex,
    int position,
    int prevVertex,
    float* pheromoneMatrix, 
    int* edgesMatrix, 
    int numberOfVertexes) 
{
    /**
        Ustalanie współczynników wagi feromonu (alpha) i heurystyki (beta)
    */
    float alpha = 1.0f;
    float beta = 3.0f;

    /**
        Pobranie kosztu krawędzi łączącej prevVertex z ownVertex
    */
        
    int edgeCost = edgesMatrix[prevVertex * numberOfVertexes + ownVertex];
    float nominator = 0.0f;

    if (edgeCost != 0) {
        nominator = (float)powf(pheromoneMatrix[prevVertex * numberOfVertexes + ownVertex], alpha) 
                    * powf(1.0f / edgeCost, beta);
    } else {
        nominator = (float)powf(pheromoneMatrix[prevVertex * numberOfVertexes + ownVertex], alpha) 
                    * powf(1.0f / 0.1f, beta);
    }


    // Zapis obliczonej wartości nominatora do tablicy chances
    chances[position] = nominator;

    /**  
    Zapis indeksu bieżącego wierzchołka do tablicy vertexes,
    aby później wiedzieć, który wierzchołek odpowiada danej wartości nominatora.
    */
    vertexes[position] = ownVertex;
}

__device__ void normalize(
    float* chances,
    float* denominator,
    int position
        ){
    chances[position] /= *denominator;
}

__global__ void findSolutions(int* solutionsPointer, int* edgesMatrix, float* pheromoneMatrix, int numberOfVertexes) {
    /**
    *   Wydzielenie współdzielonej pamięci;
    *   Obliczenie offsetu potrzebnego do poprawnego zarządzania pamiecią - w pamięci współdzielonej chcemy
    *   mieć zarówno zmienne typu int jak i float. Z powodu że typy te różnią się rozmiarem, musimy używać offsetu :))));
    *   Inicjalizacja wskaźnika do pamięci współdzielonej (przesunięcie wskaźnika sharedInt o offset);
    *   Ustawienie wskaźnika do kolejnej części pamięci współdzielonej (za tablicą sharedFloat), 
    *   która przechowuje pojedynczą wartość typu float;
    */

    extern __shared__ int sharedInt[];
    size_t shOffset = (sizeof(float)/sizeof(int)*(numberOfVertexes));
    float* sharedFloat = (float*)(&sharedInt[shOffset]);
    float* denominator = &sharedFloat[numberOfVertexes];

    // Identyfikator wątku
    int threadId = blockIdx.x;

    // Inicjalizacja stanu generatora liczb pseudolosowych
    curandState state;
    curand_init((unsigned long long)clock() + threadId, 0, 0, &state);

    /**
    *   W pamięci zarezerwowanej dla wątku o id threadId zapisujemy początkowy wierzchołek
    *   Przechowujemy go również w pamięciu współdzielonej 
    */

    int* solution = &solutionsPointer[threadId * numberOfVertexes];
    int lastVisitedVertex = (int)(curand_uniform(&state) * numberOfVertexes);
    if (lastVisitedVertex == 0) {
        lastVisitedVertex++;
    }
    solution[0] = lastVisitedVertex;
    sharedInt[0] = lastVisitedVertex;

    int visitedCount = 1;

    auto skip = false;
    auto threadIdentifier = threadIdx.x;
    if (threadIdentifier >= numberOfVertexes || threadId > numberOfVertexes) skip = true;
    auto position = threadIdentifier - 1;
    auto ownVertex = threadIdentifier;
    float alpha = 1.0f, beta = 3.0f;
    int nextVertex;
    while (visitedCount < numberOfVertexes) {
        if (threadIdentifier != 0 && !skip) {
            auto prev = sharedInt[visitedCount-1];
            if (ownVertex == prev) skip = true;
            else if (prev > ownVertex) position++;
            if (!skip) calculateNominatorToShared(sharedInt, sharedFloat, ownVertex, position, prev, pheromoneMatrix, edgesMatrix, numberOfVertexes);
        }

        /**
        *   Synchronizacja wątków po obliczeniach liczników oraz
        *   Obliczanie mianownika prawdopodobieństwa w wątku 0
        */

        __syncthreads();
        if (threadIdentifier == 0 && !skip) {
            *denominator = calculateDenominator( sharedFloat, visitedCount, numberOfVertexes);
        }
        __syncthreads(); 

        //Normalizacja prawdopodobieństwa w pozostałych wątkach
        if (threadIdentifier != 0 && !skip) {
            normalize(sharedFloat, denominator, position);
        }
        __syncthreads();

        //Wybór kolejnego wierzchołka przez wątek 0 na podstawie prawdopodobieństwa
        if (threadIdx.x == 0 && !skip) {
            nextVertex = choseVertexByProbability(sharedInt, sharedFloat, visitedCount, numberOfVertexes, state);
            sharedInt[visitedCount] = nextVertex;
            solution[visitedCount] = nextVertex;
        }
        __syncthreads();
        lastVisitedVertex = nextVertex;
        visitedCount++;

        /*if (threadIdx.x == 0 && !skip) {
            solution[visitedCount] = nextVertex;
        }*/
        
    }
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

        std::vector<int> flatEdges;
        for (const auto& row : edges) {
            flatEdges.insert(flatEdges.end(), row.begin(), row.end());
        }
        /**
            Inicjalizacja tablicy krawędzi w pamięciu GPU
        */
        int* d_edges; 
        cudaMalloc(&d_edges, flatEdges.size() * sizeof(int));
        cudaMemcpy(d_edges, flatEdges.data(), flatEdges.size() * sizeof(int), cudaMemcpyHostToDevice);

        std::vector<float> flatPheromone;
        for (const auto& row : pheromoneMatrix) {
            flatPheromone.insert(flatPheromone.end(), row.begin(), row.end());
        }
        /**
            Inicjalizacja tablicy feromonów w pamięciu GPU
        */
        float* d_pheromoneMatrix;
        cudaMalloc(&d_pheromoneMatrix, flatPheromone.size() * sizeof(float));
        cudaMemcpy(d_pheromoneMatrix, flatPheromone.data(), flatPheromone.size() * sizeof(float), cudaMemcpyHostToDevice);

        /**
            Inicjalizacja tablicy prawdopodobieństw w pamięciu GPU
        */
        float* d_probMatrix;
        cudaMalloc(&d_probMatrix, numberOfVertexes * numberOfVertexes * sizeof(float));

        /**
            Inicjalizacja tablicy mrówek w pamięciu GPU
        */
        int* d_colony;
        cudaMalloc(&d_colony, colonySize * edges.size() * sizeof(int));

        int* h_costs = (int*)malloc(colonySize * sizeof(int));
        for (auto i = 0; i < numberOfVertexes; i++) {
            h_costs[i] = i;
        }
        /**
            Inicjalizacja tablicy kosztów przejścia między miastami w pamięciu GPU
        */
        int* d_costs;
        cudaMalloc(&d_costs, colonySize * sizeof(int));

        int* h_colony = (int*)malloc(colonySize * edges.size() * sizeof(int));
        for (int j = 0; j < numberOfIterations; j++) {
            int blockMaxSize = -1;
            int threadsPerBlock = 32;
            int sharedMemorySize = blockMaxSize;
            int numberOfBlocks = colonySize / threadsPerBlock + 1;

            /**
                Wywołanie kolejnych funkcji z wykorzystaniem wątków GPU
            */
            findSolutions << <colonySize, numberOfVertexes + 1, numberOfVertexes* (sizeof(float) + sizeof(int)) + sizeof(float) >> > (d_colony, d_edges, d_pheromoneMatrix, numberOfVertexes);
            evapouratePheromoneD << <numberOfVertexes * numberOfVertexes / threadsPerBlock, threadsPerBlock >> > (d_pheromoneMatrix, 0.6, numberOfVertexes);
            leavePheromone << <numberOfBlocks, threadsPerBlock >> > (d_pheromoneMatrix, d_edges, d_colony, numberOfVertexes, 100.0, d_costs, colonySize);
            
            /**
                Skopiowanie wyników - koszty a nie trasa - do pamięci hosta
            */
            cudaMemcpy(h_costs, d_costs, colonySize * sizeof(int), cudaMemcpyDeviceToHost);
            int bestIndex = -1;
            for (auto i = 0; i < colonySize; i++) {
                if (h_costs[i] < minCost) {
                    minCost = h_costs[i];
                    bestIndex = i;
                }
            }
            if (bestIndex != -1) {
                /**
                    Zapisanie najlepszego rozwiązania do pamięci hosta
                */
                cudaMemcpy(result, &d_colony[bestIndex * numberOfVertexes], numberOfVertexes * sizeof(int), cudaMemcpyDeviceToHost);
            }
        }

        return result;
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
        for (int i = 0; i < numberOfVertexes - 2; i++)
        {
            cost += edges[solution[i]][solution[i + 1]];
            std::cout << i << ": " << cost << " ";
        }

        cost += edges[startingVertex][solution[0]];					
        cost += edges[solution[numberOfVertexes-2]][startingVertex];

        return cost;
    }

    void ACOImplementation::initializePheromoneMatrix(int aproximatedSolutionCost)
    {
        float tau_zero = 10000.0 / (float)aproximatedSolutionCost;
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
        int* solution = new int[numberOfVertexes-1];

        int randIndexI, randIndexJ;

        for (int i = 0; i < numberOfVertexes-1; i++) solution[i] = i+1;

        for (int i = 0; i < numberOfVertexes - 1 ; i++)
        {
            randIndexI = rand() % (numberOfVertexes -1);
            randIndexJ = rand() % (numberOfVertexes -1);
            std::swap(solution[randIndexI], solution[randIndexJ]);
        }

        return calculateSolutionCost(solution) * 0.6f;
    }
}

