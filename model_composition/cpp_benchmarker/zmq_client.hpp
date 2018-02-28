#ifndef CLIPPER_ZMQ_CLIENT_HPP
#define CLIPPER_ZMQ_CLIENT_HPP

#include <mutex>

// #include <folly/ProducerConsumerQueue.h>
#include <concurrentqueue.h>
#include <clipper/callback_threadpool.hpp>
#include <clipper/datatypes.hpp>
#include <zmq.hpp>

namespace zmq_client {

using namespace clipper;

const std::string LOGGING_TAG_ZMQ_CLIENT = "ZMQ_FRONTEND";

// We may have up to 50,000 outstanding requests
constexpr size_t RESPONSE_QUEUE_SIZE = 100000;
constexpr size_t NUM_REQUESTS_RECV = 100;
constexpr size_t NUM_RESPONSES_SEND = 1000;

constexpr size_t TOTAL_DATA_BYTES = 299 * 299 * 3 * sizeof(float) * RESPONSE_QUEUE_SIZE;

// Tuple of input, request id, client id
typedef std::tuple<InputVector, int, int> FrontendRPCRequest;
// Tuple of output, request id, client id. Request id and client ids
// should match corresponding ids of a FrontendRPCRequest object
typedef std::tuple<Output, int, int> FrontendRPCResponse;

class FrontendRPCClient {
 public:
  FrontendRPCClient();
  ~FrontendRPCClient();

  FrontendRPCClient(const FrontendRPCClient &) = delete;
  FrontendRPCClient &operator=(const FrontendRPCClient &) = delete;

  void start(const std::string address, int send_port, int recv_port);
  void stop();
  void send_request(InputVector input);

 private:
  void manage_send_service(const std::string ip, int port);
  void manage_recv_service(const std::string ip, int port);
  void shutdown_service(zmq::socket_t &socket);
  void recv_responses(zmq::socket_t &socket, size_t num_responses);

  std::shared_ptr<moodycamel::ConcurrentQueue<InputVector>>
      request_queue_;
  // std::shared_ptr<clipper::CallbackThreadPool> prediction_executor_;
  std::atomic_bool active_;
  std::thread rpc_send_thread_;
  std::thread rpc_recv_thread_;

  std::shared_ptr<metrics::Meter> request_enqueue_meter_;

  std::shared_ptr<metrics::Meter> response_enqueue_meter_;
  std::shared_ptr<metrics::Meter> response_dequeue_meter_;
  // std::shared_ptr<metrics::Histogram> malloc_latency_;
  std::shared_ptr<metrics::Histogram> recv_latency_;

};

}  // namespace zmq_client

#endif  // CLIPPER_ZMQ_CLIENT_HPP
