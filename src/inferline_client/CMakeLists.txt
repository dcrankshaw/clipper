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

add_executable(contention_driver
        src/contention_driver.cpp
        src/driver.cpp
        src/zmq_client.cpp)

target_link_libraries(contention_driver
        clipper
        boost
        cxxopts
        httpserver
        zmqcpp
        base64json
)

add_executable(resnet_cascade
        src/resnet_cascade.cpp
        src/driver.cpp
        src/zmq_client.cpp)

target_link_libraries(resnet_cascade
        clipper
        boost
        cxxopts
        httpserver
        zmqcpp
        base64json
)

add_executable(text_driver_one
        src/text_driver_one.cpp
        src/driver.cpp
        src/zmq_client.cpp)

target_link_libraries(text_driver_one
        clipper
        boost
        cxxopts
        httpserver
        zmqcpp
        base64json
)
