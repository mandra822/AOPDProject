# setting up sources
add_library(gpu_module
    src/ACOImplementation.cu)
set_target_properties(gpu_module PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
# setting up include directories
target_include_directories(gpu_module
    PUBLIC include
    PRIVATE src)

# linking against the desired targets
target_link_libraries(gpu_module
    PRIVATE
    project_options
    project_warnings
    utils_module)

install(TARGETS gpu_module
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${AODPProject}/
    RUNTIME DESTINATION       ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION       ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION      ${CMAKE_INSTALL_INCLUDEDIR}/${AODPProject}/
    ARCHIVE DESTINATION       ${CMAKE_INSTALL_SHAREDSTATEDIR})
