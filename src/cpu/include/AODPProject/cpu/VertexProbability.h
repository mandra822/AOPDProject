#pragma once
#include <vector>

namespace CPU {
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
}

