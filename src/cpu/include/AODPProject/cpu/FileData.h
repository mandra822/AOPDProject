#pragma once
#include <string>
#include <vector>

using namespace std;

struct FileData {
	std::string fileName;
	int numberOfIterations;
	int colonySize;
};

class FileManager {
public:
	void loadFromFile(vector<vector<int>>& edges, int& numberOfVertexes, string fileName);
	vector<FileData> loadTestInitFile(string initFileName);
};