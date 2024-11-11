#define CONFIG_CATCH_MAIN
#include <catch2/catch_all.hpp>

#include "AODPProject/utils/FileData.h"
#include "AODPProject/cpu/ACOImplementation.h"
#include <cmath>

TEST_CASE( "Basic test", "[basic]" ) {
    srand(time(NULL));
    FileManager fileManager {};
    auto testFiles = fileManager.loadTestInitFile("instances/test_file_setup.txt");
    for (auto file : testFiles) {
        int numberOfVertexes;
        std::vector<std::vector<int> > edges {}; 
        fileManager.loadFromFile(edges, numberOfVertexes, file.fileName);

        int startingVertex = 0;  // Zaczynamy od wierzcho³ka 0

        float alpha = 1.5;    // Wp³yw feromonów
        float beta = 3.0;     // Wp³yw widocznoœci (odwrotnoœci kosztu)
        ACOImplementation aco;
        aco.init(startingVertex, edges, alpha, beta, numberOfVertexes, file.colonySize);

        vector<int> result = aco.runAcoAlgorith(file.numberOfIterations);
        REQUIRE( std::abs((aco.calculateSolutionCost(result) - file.bestSolution) / file.bestSolution) < file.acceptedError);
    }

    REQUIRE( 1 == 1 );
    REQUIRE( 6 == 2 );
}
