#pragma once
#include <vector>

using namespace std;
namespace NONCONCURRENT {
    class Ant
    {	

    public:
        vector<int> visitedVertexes;
        void addNewVertex(int vertexId);
    };
}
