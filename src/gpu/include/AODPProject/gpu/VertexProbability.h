#pragma once
#include <vector>

namespace GPU {
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
