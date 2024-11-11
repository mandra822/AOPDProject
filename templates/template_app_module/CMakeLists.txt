# setting up sources
add_library(MODULE_NAME_module
    src/main.cpp)

# setting up include directories
target_include_directories(MODULE_NAME_module
    PUBLIC include
    PRIVATE src)

# linking against the desired targets
target_link_libraries(MODULE_NAME_module
    PRIVATE
    project_options
    project_warnings)

install(TARGETS MODULE_NAME_module
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/
    RUNTIME DESTINATION       ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION       ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION      ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/
    ARCHIVE DESTINATION       ${CMAKE_INSTALL_SHAREDSTATEDIR})
