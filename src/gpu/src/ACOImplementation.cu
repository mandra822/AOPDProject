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
#include <climits>
#include <cmath>
/**
    Funkcja sprawdzająca czy czy wywołanie danej funkcji CUDA zakończyło się sukcesem
    Jeśli nie, program kończy działanie
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void evapouratePheromoneD(float* pheromoneMatrix, float rate, int numberOfVertexes) {
    int threadId = threadIdx.x + blockDim.x * blockIdx.x; // Obliczanie ID wątku
    if (threadId >= numberOfVertexes * numberOfVertexes) return;
    pheromoneMatrix[threadId] *= rate;
}

__global__ void leavePheromone(float* pheromoneMatrix, int* edgesMatrix, int* ants, int solutionSize, float Qcycl, int* costs, int colonySize, int numberOfVertexes) {
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId >= colonySize) return;

    int cost = 0;

    for (int i = 0; i < solutionSize - 1; i++) {
        int sourceVertex = ants[threadId * solutionSize + i];
        int targetVertex = ants[threadId * solutionSize + i + 1];
        cost += edgesMatrix[sourceVertex * numberOfVertexes + targetVertex];
    }
    costs[threadId] = cost; // Zapisanie kosztu do pamięci globalnej.

    for (int i = 0; i < solutionSize - 1; i++) {
        int sourceVertex = ants[threadId * solutionSize + i];
        int targetVertex = ants[threadId * solutionSize + i + 1];
        pheromoneMatrix[sourceVertex * numberOfVertexes + targetVertex] += Qcycl / (float)cost;
    }
}

/**
    Funckja służąca do wybrania wierzchołka na podstawie prawdopodobieństwa
 */
__device__ int chooseVertexByProbability(
    float* chances, int numberOfVertexes, curandState& state) {

    float toss = curand_uniform(&state); // Losowanie liczby z zakresu [0, 1).
    float cumulative = 0.0f;

    for (int i = 1; i < numberOfVertexes; i++) { //pomijamy miasto 0
        cumulative += chances[i];
        if (cumulative >= toss) {
            return i; // Zwraca wierzchołek gdy suma przekroczy wartość toss
        }
    }

    return numberOfVertexes - 1; // jakby coś poszło nie tak, zwracamy ostatni wierzchołek
}

/**
    Funkcja CUDA służąca do znajdowania rozwiązania przez mrówki
 */
__global__ void findSolutions(int* solutionsPointer, int* edgesMatrix, float* pheromoneMatrix,
    int numberOfVertexes, float alpha, float beta, int startingVertex) {

    int antId = blockIdx.x; // jednej mrówce w algorytmie odpowiada jeden blok
    int threadId = threadIdx.x; // pobieramy id wątku w ramach bloku

    int solutionSize = numberOfVertexes - 1;    // ponieważ ustalono że rozwiązanie nie zawiera miasta początkowego, 
    // rozmiar rozwiązania musi być o 1 mniejszy niż liczba miast 

    // zadeklarowanie pamięci współdzielonej dla każdego bloku
    extern __shared__ unsigned char sharedMem[];
    int* visited = (int*)sharedMem;
    float* chances = (float*)&visited[numberOfVertexes];

    for (int i = threadId; i < numberOfVertexes; i += blockDim.x) {
        visited[i] = false;
    }

    __syncthreads();

    // nie chcemy odwiedzać miasta 0 więc zaznaczamy, że był już odwiedzony
    if (threadId == 0) {
        visited[0] = true;
    }
    __syncthreads();

    /**
        Inicjalizacja generatora liczb (pseudo)losowych;
        Dzięki dodaniu clock() antId, każda mrówka używa innego ziarna generatora;
     */

    curandState state;
    curand_init((unsigned long long)clock() + antId, threadId, 0, &state);

    int* solution = &solutionsPointer[antId * solutionSize];

    if (startingVertex == 0) {
        startingVertex = 1;
    }

    if (threadId == 0) {
        solution[0] = startingVertex;
        visited[startingVertex] = true;
    }

    int visitedCount = 1;
    __syncthreads();

    // pętla tworząca trasę mrówki
    while (visitedCount < solutionSize) {
        if (threadId < numberOfVertexes) {
            if (visited[threadId]) {
                chances[threadId] = 0.0f; // wyzerowujemy szanse dla odwiedzonych
            }
            else {
                int prev = solution[visitedCount - 1];
                int edgeCost = edgesMatrix[prev * numberOfVertexes + threadId];

                // obliczenie szansy dla danego wątku
                float costFactor = (edgeCost > 0) ? (1.0f / edgeCost) : (1.0f / 0.1f);
                float tau = pheromoneMatrix[prev * numberOfVertexes + threadId];
                float nominator = powf(tau, alpha) * powf(costFactor, beta);

                chances[threadId] = nominator;
            }
        }
        __syncthreads();

        // Dla wątku o id = 0, normalizujemy szanse i wybieramy następne miasto
        if (threadId == 0) {
            float sum = 0.0f;
            for (int i = 1; i < numberOfVertexes; i++) {
                sum += chances[i];
            }

            if (sum > 0.0f) {
                for (int i = 1; i < numberOfVertexes; i++) {
                    chances[i] /= sum;
                }

                chances[0] = 0.0f;
            }
            else {
                // awaryjny scenariusz jeśli suma szans wynosi 0
                int idx = -1;
                for (int i = 1; i < numberOfVertexes; i++) {
                    if (!visited[i]) {
                        idx = i;
                        break;
                    }
                }
                for (int i = 1; i < numberOfVertexes; i++) {
                    chances[i] = (i == idx) ? 1.0f : 0.0f;
                }
                chances[0] = 0.0f;
            }
        }
        __syncthreads();

        // dodanie wierzchołka do rozwiązania
        int nextVertex = -1;
        if (threadId == 0) {
            nextVertex = chooseVertexByProbability(chances, numberOfVertexes, state);
            solution[visitedCount] = nextVertex;
            visited[nextVertex] = true;
        }
        __syncthreads();

        visitedCount++;
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
        this->result = (int*)malloc((numberOfVertexes - 1) * sizeof(int));
        this->minCost = INT_MAX;
        initializePheromoneMatrix((int)calculateApproximatedSolutionCost());
    }

    int* ACOImplementation::runAcoAlgorith(int numberOfIterations)
    {
        int solutionSize = numberOfVertexes - 1;

        // Zapis macierzy kosztów do gpu
        std::vector<int> flatEdges;
        flatEdges.reserve(numberOfVertexes * numberOfVertexes);
        for (const auto& row : edges) {
            flatEdges.insert(flatEdges.end(), row.begin(), row.end());
        }

        int* d_edges;
        gpuErrchk(cudaMalloc(&d_edges, flatEdges.size() * sizeof(int)));
        gpuErrchk(cudaMemcpy(d_edges, flatEdges.data(), flatEdges.size() * sizeof(int), cudaMemcpyHostToDevice));

        // Zapis macierzy feromonu do gpu
        std::vector<float> flatPheromone;
        flatPheromone.reserve(numberOfVertexes * numberOfVertexes);
        for (const auto& row : pheromoneMatrix) {
            flatPheromone.insert(flatPheromone.end(), row.begin(), row.end());
        }

        float* d_pheromoneMatrix;
        gpuErrchk(cudaMalloc(&d_pheromoneMatrix, flatPheromone.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(d_pheromoneMatrix, flatPheromone.data(), flatPheromone.size() * sizeof(float), cudaMemcpyHostToDevice));

        // inicjalizacja rozwiązań mrówek
        int* d_colony;
        gpuErrchk(cudaMalloc(&d_colony, colonySize * solutionSize * sizeof(int)));

        int* h_costs = (int*)malloc(colonySize * sizeof(int));
        int* d_costs;
        gpuErrchk(cudaMalloc(&d_costs, colonySize * sizeof(int)));

        // Obliczenie rozmiaru shared memory 
        size_t sharedMemSize = numberOfVertexes * sizeof(int) + numberOfVertexes * sizeof(float);

        int threadsPerBlock = numberOfVertexes;
        int blocks = colonySize;

        for (int j = 0; j < numberOfIterations; j++) {
            // Stworzenie rozwiązań
            findSolutions << <blocks, threadsPerBlock, sharedMemSize >> > (d_colony, d_edges, d_pheromoneMatrix,
                numberOfVertexes, alpha, beta, startingVertex);
            gpuErrchk(cudaDeviceSynchronize());

            // Odparowanie feromonu
            int totalEdges = numberOfVertexes * numberOfVertexes;
            int evapBlocks = (totalEdges + threadsPerBlock - 1) / threadsPerBlock;
            evapouratePheromoneD << <evapBlocks, threadsPerBlock >> > (d_pheromoneMatrix, 0.6f, numberOfVertexes);
            gpuErrchk(cudaDeviceSynchronize());

            // Aktualizacja feromonu
            int pheromoneBlocks = (colonySize + threadsPerBlock - 1) / threadsPerBlock;
            leavePheromone << <pheromoneBlocks, threadsPerBlock >> > (d_pheromoneMatrix, d_edges, d_colony, solutionSize, 100.0f, d_costs, colonySize, numberOfVertexes);
            gpuErrchk(cudaDeviceSynchronize());

            /**
                Skopiowanie wyników - koszty a nie trasa - do pamięci hosta
            */
            gpuErrchk(cudaMemcpy(h_costs, d_costs, colonySize * sizeof(int), cudaMemcpyDeviceToHost));
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
                gpuErrchk(cudaMemcpy(result, &d_colony[bestIndex * solutionSize], solutionSize * sizeof(int), cudaMemcpyDeviceToHost));
            }
        }

        // zwolnienie pamięci
        free(h_costs);
        cudaFree(d_colony);
        cudaFree(d_pheromoneMatrix);
        cudaFree(d_edges);
        cudaFree(d_costs);

        return result;
    }

    /**
        Funkcja obliczająca koszt rozwiązania
    */
    int ACOImplementation::calculateSolutionCost(int* solution)
    {
        int cost = 0;
        for (int i = 0; i < numberOfVertexes - 2; i++) {
            cost += edges[solution[i]][solution[i + 1]];
        }
        cost += edges[startingVertex][solution[0]];
        cost += edges[solution[numberOfVertexes - 2]][startingVertex];
        return cost;
    }

    void ACOImplementation::initializePheromoneMatrix(int approximatedSolutionCost)
    {
        float tau_zero = colonySize / (float)approximatedSolutionCost;
        pheromoneMatrix.assign(numberOfVertexes, std::vector<float>(numberOfVertexes, tau_zero));
    }

    float ACOImplementation::calculateApproximatedSolutionCost()
    {
        int solutionSize = numberOfVertexes - 1;
        int* solution = new int[solutionSize];

        for (int i = 0; i < solutionSize; i++) solution[i] = i + 1;

        for (int i = 0; i < solutionSize; i++)
        {
            int randIndexI = rand() % solutionSize;
            int randIndexJ = rand() % solutionSize;
            std::swap(solution[randIndexI], solution[randIndexJ]);
        }

        float approxCost = (float)calculateSolutionCost(solution) * 0.6f;
        delete[] solution;
        return approxCost;
    }

}
