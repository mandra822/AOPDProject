#define CONFIG_CATCH_MAIN
#include <catch2/catch_all.hpp>

#include "AODPProject/utils/FileData.h"
#include "AODPProject/utils/TimeUtils.h"
#include "AODPProject/cpu/ACOImplementation.h"
#include <cmath>
#include <iostream>

TEST_CASE( "Basic test", "[basic][!mayfail]" ) {
    srand(time(NULL));
    FileManager fileManager {};
    auto testFiles = fileManager.loadTestInitFile("instances/test_file_setup.txt");
    for (auto file : testFiles) {
        SECTION( "ACO for instance from " + file.fileName ) {
            GIVEN("Given data from file " + file.fileName) {
                int numberOfVertexes;
                std::vector<std::vector<int> > edges {}; 
                fileManager.loadFromFile(edges, numberOfVertexes, file.fileName);

                int startingVertex = 0;  // Zaczynamy od wierzcho³ka 0

                float alpha = 1.5;    // Wp³yw feromonów
                float beta = 3.0;     // Wp³yw widocznoœci (odwrotnoœci kosztu)
                int colonySize = file.colonySize;
                int numberOfIterations = file.numberOfIterations;
                ACOImplementation aco;
                vector<int> result; 
                WHEN("Runbning ACO") {
                    auto elapsedMiliseconds = TimeUtils::elapsedEvaluatingFunc([startingVertex, edges, alpha, beta, numberOfVertexes, colonySize, &aco, numberOfIterations, &result]() { 
                        aco.init(startingVertex, edges, alpha, beta, numberOfVertexes, colonySize);
                        result = aco.runAcoAlgorith(numberOfIterations);
                    });
                    THEN("Getting results") {
                        std::cout<<"\nGot solution for file " << file.fileName << "\nBest solution: " << file.bestSolution << "\nComputed solution: "<< aco.calculateSolutionCost(result) << "\nIn time: " << elapsedMiliseconds.count() << " ms\n";
                        auto error = std::abs(static_cast<float>(aco.calculateSolutionCost(result) - file.bestSolution) / file.bestSolution);
                        std::cout<< "Error: " << error << "\nAccepted error: " << file.acceptedError<< '\n';
                        REQUIRE( error  < file.acceptedError);
                    }
                }
            }
        }
    }
}
