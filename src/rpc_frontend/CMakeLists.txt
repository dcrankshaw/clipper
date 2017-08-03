cmake_minimum_required(VERSION 3.2 FATAL_ERROR)




######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################

INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIRS})


# # gRPC
# include(FindGRPC)
# find_package(g REQUIRED)
# # message(STATUS "Using gRPC ${gRPC_VERSION}")

# gRPC C++ plugin
# get_target_property(gRPC_CPP_PLUGIN_EXECUTABLE gRPC::grpc_cpp_plugin
#     IMPORTED_LOCATION_RELEASE)

# TODO: need to add a custom target to make sure grpc_cpp_plugin gets
# built first?
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

# Proto file
get_filename_component(predict_proto "src/protos/clipper_frontend.proto" ABSOLUTE)
get_filename_component(predict_proto_path "${predict_proto}" PATH)

# Generated sources
protobuf_generate_cpp(predict_proto_srcs predict_proto_hdrs "${predict_proto}")
set(predict_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/clipper_frontend.grpc.pb.cc")
set(predict_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/clipper_frontend.grpc.pb.h")
add_custom_command(
      OUTPUT "${predict_grpc_srcs}" "${predict_grpc_hdrs}"
      COMMAND protoc
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}" -I "${predict_proto_path}"
      --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}"
        "${predict_proto}"
      DEPENDS "${predict_proto}")

# Generated include directory
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_executable(clipper_grpc_frontend
    src/grpc_frontend_main.cpp
    ${predict_proto_srcs}
    ${predict_grpc_srcs})

target_link_libraries(clipper_grpc_frontend
    ${PROTOBUF_LIBRARIES}
    grpc++_unsecure
    clipper
    boost
    # httpserver
    cxxopts
    # rapidjson
    )


# # Targets greeter_[async_](client|server)
# foreach(_target
#   predict_client predict_server)
#   add_executable(${_target} "${_target}.cc"
#     ${predict_proto_srcs}
#     ${predict_grpc_srcs})
#   target_link_libraries(${_target}
#     protobuf::libprotobuf
#     gRPC::grpc++_unsecure)
# endforeach()


######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################



##################################
# Query REST frontend executable

# add_executable(gprc_frontend src/grpc_frontend_main.cpp)
#
# target_link_libraries(grpc_frontend clipper boost httpserver cxxopts rapidjson)

