find_package(Catch2 REQUIRED)
add_executable(main_module_test
        main_module_test.cpp
    )
target_link_libraries(main_module_test#${PROJECT_NAME}_alisp_test
    PRIVATE
    project_options
    project_warnings
    utils_module
    cpu_module
    gpu_module
    nonconcurrent_module
    Catch2::Catch2WithMain
    )
target_include_directories(main_module_test#${PROJECT_NAME}_alisp_test
    PUBLIC ../include
    PRIVATE ../src
    )


include(CTest)
include(Catch)
catch_discover_tests(main_module_test)
