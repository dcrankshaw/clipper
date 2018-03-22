cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_executable(profiler
        src/profiler.cpp
        src/driver.cpp
        src/zmq_client.cpp)

target_link_libraries(profiler
        clipper
        boost
        cxxopts
        httpserver
        zmqcpp
        base64json
)

add_executable(image_driver_one
        src/image_driver_one.cpp
        src/driver.cpp
        src/zmq_client.cpp)

target_link_libraries(image_driver_one
        clipper
        boost
        cxxopts
        httpserver
        zmqcpp
        base64json
)

add_executable(image_driver_two
        src/image_driver_two.cpp
        src/driver.cpp
        src/zmq_client.cpp)

target_link_libraries(image_driver_two
        clipper
        boost
        cxxopts
        httpserver
        zmqcpp
        base64json
)
