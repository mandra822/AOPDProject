#include "AODPProject/nonconcurrent/Ant.h"

namespace NONCONCURRENT {
    void Ant::addNewVertex(int vertexId)
    {
        visitedVertexes.push_back(vertexId);
    }
}
