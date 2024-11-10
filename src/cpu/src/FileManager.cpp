#include "FileData.h"

void FileManager::loadFromFile(vector<vector<int>>& edges, int& numberOfVertexes, string fileName) {
    ifstream file(fileName);
    if (!file) {
        cerr << "Error opening file!" << endl;
        return;
    }
    file >> numberOfVertexes;
    edges.resize(numberOfVertexes);
    for (int i = 0; i < numberOfVertexes; i++) {
        edges[i].resize(numberOfVertexes);
        for (int j = 0; j < numberOfVertexes; j++) {
            file >> edges[i][j];
        }
    }
}

vector<FileData> FileManager::loadTestInitFile(string initFileName) {
    ifstream file(initFileName);
    if (!file) {
        cerr << "Error opening init file!" << endl;
        return;
    }
    vector<FileData> testFiles;
    int numberOfFiles;
    file >> numberOfFiles;
    testFiles.reserve(numberOfFiles);
    for (int i = 0; i < numberOfFiles; i++) {
        FileData testFile;
        file >> testFile.fileName;
        file >> testFile.colonySize;
        file >> testFile.numberOfIterations;
        testFiles.push_back(testFile);
    }
    return testFiles;
}