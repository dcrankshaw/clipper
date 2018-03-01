cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_executable(profiler
        src/profiler.cpp
        src/predictor.cpp
        src/zmq_client.cpp)

target_link_libraries(profiler
        clipper
        boost
        cxxopts
        httpserver
        zmqcpp
        base64json
)
