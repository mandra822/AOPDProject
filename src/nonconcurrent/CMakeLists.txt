# setting up sources
add_library(nonconcurrent_module
    src/ACOImplementation.cpp
    src/Ant.cpp
    )

# setting up include directories
target_include_directories(nonconcurrent_module
    PUBLIC include
    PRIVATE src)

# linking against the desired targets
target_link_libraries(nonconcurrent_module
    PRIVATE
    project_options
    project_warnings)

install(TARGETS nonconcurrent_module
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${AODPProject}/
    RUNTIME DESTINATION       ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION       ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION      ${CMAKE_INSTALL_INCLUDEDIR}/${AODPProject}/
    ARCHIVE DESTINATION       ${CMAKE_INSTALL_SHAREDSTATEDIR})
