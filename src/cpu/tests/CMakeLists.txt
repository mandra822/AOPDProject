find_package(Catch2 REQUIRED)
add_executable(cpu_module_test
        cpu_module_test.cpp
    )
message(STATUS "Include directories: ${CMAKE_INCLUDE_DIRECTORIES} [::::>] ${Catch2_INCLUDE_DIRS}")
target_link_libraries(cpu_module_test#${PROJECT_NAME}_alisp_test
    PRIVATE
    project_options
    project_warnings
    Catch2::Catch2WithMain
    )
target_include_directories(cpu_module_test#${PROJECT_NAME}_alisp_test
    PUBLIC ../include
    PRIVATE ../src
    )


include(CTest)
include(Catch)
catch_discover_tests(cpu_module_test)
