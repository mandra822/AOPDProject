# setting up sources
add_executable(cpu_module
    src/main.cpp
    src/ACOImplementation.cpp
    src/Ant.cpp
    )

# setting up include directories
target_include_directories(cpu_module
    PUBLIC include
    PRIVATE src)

# linking against the desired targets
target_link_libraries(cpu_module
    PRIVATE
    project_options
    project_warnings
    utils_module)

install(TARGETS cpu_module
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/
    RUNTIME DESTINATION       ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION       ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION      ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/
    ARCHIVE DESTINATION       ${CMAKE_INSTALL_SHAREDSTATEDIR})

if(ENABLE_TESTING)
    #    enable_testing()
    add_subdirectory(tests)
endif()
