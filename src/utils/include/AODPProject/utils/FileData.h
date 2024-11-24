#pragma once
#include <string>
#include <vector>


struct FileData {
	std::string fileName;
	int numberOfIterations;
	int colonySize;
    int bestSolution;
    float acceptedError;
};

class FileManager {
public:
	void loadFromFile(std::vector<std::vector<int> >& edges, int& numberOfVertexes, std::string fileName);
    std::vector<FileData> loadTestInitFile(std::string initFileName);
};
