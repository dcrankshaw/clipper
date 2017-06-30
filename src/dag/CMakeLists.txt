cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

##################################
# Benchmark executable

add_executable(dag_main src/dag_main.cpp)

target_link_libraries(dag_main clipper boost cxxopts)

