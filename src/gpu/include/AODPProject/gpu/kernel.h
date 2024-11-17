#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>

__global__ void findSolutions(int* solutionsPointer, float* pheromoneMatrix, int* edgesMatrix, int numberOfVertexes);