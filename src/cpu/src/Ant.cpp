#include "AODPProject/cpu/Ant.h"

namespace CPU {
    void Ant::addNewVertex(int vertexId)
    {
        visitedVertexes.push_back(vertexId);
    }
}
