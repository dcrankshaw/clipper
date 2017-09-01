#include "zmq_frontend.hpp"

#include <clipper/config.hpp>
#include <clipper/constants.hpp>
#include <clipper/query_processor.hpp>
#include <cxxopts.hpp>

int main(int argc, char* argv[]) {
  cxxopts::Options options("grpc_frontend",
                           "Clipper query processing frontend");
  // clang-format off
  options.add_options()
      ("redis_ip", "Redis address",
       cxxopts::value<std::string>()->default_value("localhost"))
      ("redis_port", "Redis port",
       cxxopts::value<int>()->default_value("6379"))
      ("num_rpc_threads_size", "Number of threads for the grpc frontend",
       cxxopts::value<int>()->default_value("2"))
      ("rpc_recv_max", "", cxxopts::value<int>()->default_value("1"))
      ("rpc_send_max", "", cxxopts::value<int>()->default_value("-1"));
  // clang-format on
  options.parse(argc, argv);

  clipper::Config& conf = clipper::get_config();
  conf.set_redis_address(options["redis_ip"].as<std::string>());
  conf.set_redis_port(options["redis_port"].as<int>());
  conf.set_rpc_max_recv(options["rpc_recv_max"].as<int>());
  conf.set_rpc_max_send(options["rpc_send_max"].as<int>());
  // conf.set_task_execution_threadpool_size(options["threadpool_size"].as<int>());
  conf.ready();

  zmq_frontend::ServerImpl server("0.0.0.0", 1337);

  using namespace std::chrono_literals;
  while (true) {
    std::this_thread::sleep_for(10s);
    server.get_metrics();
    std::cout << server.get_metrics() << std::endl;
  }
}