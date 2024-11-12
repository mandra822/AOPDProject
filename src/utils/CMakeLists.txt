# setting up sources
add_library(utils_module
    src/FileManager.cpp
    )

# setting up include directories
target_include_directories(utils_module
    PUBLIC include
    PRIVATE src)

# linking against the desired targets
target_link_libraries(utils_module
    PRIVATE
    project_options
    project_warnings)

install(TARGETS utils_module
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${AODPProject}/
    RUNTIME DESTINATION       ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION       ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION      ${CMAKE_INSTALL_INCLUDEDIR}/${AODPProject}/
    ARCHIVE DESTINATION       ${CMAKE_INSTALL_SHAREDSTATEDIR})
