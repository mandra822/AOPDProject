#pragma once
#include <vector>

using namespace std;

struct VertexProbability {
	int vertex;
	float probability;
};

struct VertexProbabilityGreater	
{
	bool operator()(const VertexProbability& lx, const VertexProbability& rx) const {
		return lx.probability > rx.probability;
	}
};

