#ifndef CLIPPER_FRONTEND_RPC_SERVICE_HPP
#define CLIPPER_FRONTEND_RPC_SERVICE_HPP

#include <mutex>

#include <zmq.hpp>
#include <clipper/datatypes.hpp>
#include <folly/ProducerConsumerQueue.h>
#include <wangle/concurrent/CPUThreadPoolExecutor.h>

namespace zmq_frontend {

using namespace clipper;

const std::string LOGGING_TAG_ZMQ_FRONTEND = "ZMQ_FRONTEND";

// We may have up to 50,000 outstanding requests
constexpr size_t RESPONSE_QUEUE_SIZE = 50000;
constexpr size_t NUM_REQUESTS_RECV = 100;
constexpr size_t NUM_RESPONSES_SEND = 100;

// Pair of input and request id
typedef std::pair<std::shared_ptr<Input>, int> FrontendRPCRequest;
// Pair of output and request id. Request id should match the id of a FrontendRPCRequest object
typedef std::pair<Output, int> FrontendRPCResponse;

class FrontendRPCService {
 public:

  FrontendRPCService();
  ~FrontendRPCService();

  FrontendRPCService(const FrontendRPCService &) = delete;
  FrontendRPCService &operator=(const FrontendRPCService &) = delete;

  void start(const std::string address, int port);
  void stop();
  void send_response(FrontendRPCResponse response);
  void add_application(std::string name, std::function<void(FrontendRPCRequest)> app_function);

 private:
  void manage_service(const std::string ip, int port);
  void shutdown_service(zmq::socket_t& socket);
  void receive_request(zmq::socket_t &socket,
                       std::unordered_map<size_t, const std::vector<uint8_t>>& outstanding_requests,
                       size_t& request_id);
  void send_responses(zmq::socket_t &socket,
                      std::unordered_map<size_t, const std::vector<uint8_t>>& outstanding_requests);

  std::mutex response_queue_insertion_mutex_;
  std::shared_ptr<folly::ProducerConsumerQueue<FrontendRPCResponse>> response_queue_;
  std::shared_ptr<wangle::CPUThreadPoolExecutor> prediction_executor_;
  std::atomic_bool active_;
  std::mutex app_functions_mutex_;
  // Mapping from app name to prediction function
  std::unordered_map<std::string, std::function<void(FrontendRPCRequest)>> app_functions_;
  std::thread rpc_thread_;
};

} // namespace zmq_frontend









#endif //CLIPPER_FRONTEND_RPC_SERVICE_HPP
