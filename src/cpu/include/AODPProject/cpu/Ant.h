#pragma once
#include <vector>

using namespace std;

namespace CPU {
    class Ant
    {	

    public:
        vector<int> visitedVertexes;
        void addNewVertex(int vertexId);
    };
}
