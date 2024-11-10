#include "AODPProject/utils/FileData.h"
#include <fstream>
#include <iostream>

void FileManager::loadFromFile(std::vector<std::vector<int> >& edges, int& numberOfVertexes, std::string fileName) {
    std::ifstream file(fileName, std::ios::in);
    if (!file) {
        std::cerr << "Error opening file!" << std::endl;
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

std::vector<FileData> FileManager::loadTestInitFile(std::string initFileName) {
    std::ifstream file(initFileName, std::ios::in);
    if (!file) {
        std::cerr << "Error opening init file!" << std::endl;
        return {};
    }
    std::vector<FileData> testFiles;
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
