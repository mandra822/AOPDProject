# setting up sources
add_library(main_module
    src/main.cpp)

# setting up include directories
target_include_directories(main_module
    PUBLIC include
    PRIVATE src)

# linking against the desired targets
target_link_libraries(main_module
    PRIVATE
    project_options
    project_warnings)

install(TARGETS main_module
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${AODPProject}/
    RUNTIME DESTINATION       ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION       ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION      ${CMAKE_INSTALL_INCLUDEDIR}/${AODPProject}/
    ARCHIVE DESTINATION       ${CMAKE_INSTALL_SHAREDSTATEDIR})

if(ENABLE_TESTING)
    #    enable_testing()
    add_subdirectory(tests)
endif()
