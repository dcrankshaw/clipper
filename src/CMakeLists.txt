cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project("Clipper")

# Targets that we develop
add_subdirectory(libs)
add_subdirectory(libclipper)
# add_subdirectory(rest_frontend)
# add_subdirectory(rpc_frontend)
add_subdirectory(zmq_frontend)
add_subdirectory(management)
# add_subdirectory(benchmarks)
add_subdirectory(container)
