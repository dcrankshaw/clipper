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
constexpr size_t QUEUE_SIZE = 100000;
constexpr size_t NUM_REQUESTS_RECV = 100;
constexpr size_t NUM_RESPONSES_SEND = 1000;

constexpr size_t TOTAL_DATA_BYTES =
    299 * 299 * 3 * sizeof(float) * RESPONSE_QUEUE_SIZE;

// Tuple of request ID, app name, input
typedef std::tuple<int, std::string, ClientFeatureVector>
    FrontendRPCClientRequest;
// Tuple of request id, output.
// typedef std::tuple<int, Output> FrontendRPCClientResponse;

class FrontendRPCClient {
 public:
  FrontendRPCClient();
  explicit FrontendRPCClient(int num_threads);
  ~FrontendRPCClient();

  FrontendRPCClient(const FrontendRPCClient &) = delete;
  FrontendRPCClient &operator=(const FrontendRPCClient &) = delete;

  void start(const std::string address, int send_port, int recv_port);
  void stop();
  void send_request(std::string app_name, ClientFeatureVector input,
                    std::function<void(ClientFeatureVector)> callback);

 private:
  void manage_send_service(const std::string ip, int port);
  void manage_recv_service(const std::string ip, int port);
  void recv_response(zmq::socket_t &socket);
  void handle_new_connection(zmq::socket_t &socket);

  std::shared_ptr<moodycamel::ConcurrentQueue<FrontendRPCClientRequest>>
      request_queue_;
  // std::shared_ptr<clipper::CallbackThreadPool> prediction_executor_;
  std::atomic_bool active_;
  std::thread rpc_send_thread_;
  std::thread rpc_recv_thread_;

  std::unordered_map<int, std::function<void(ClientFeatureVector)>>
      closure_map_;
  std::mutex closure_map_mutex_;

  CallbackThreadPool closure_threadpool_;

  int client_id_;
  int std::atomic<int> request_id_;
  std::atomic_bool connected_;

  // std::shared_ptr<metrics::Meter> request_enqueue_meter_;
  //
  // std::shared_ptr<metrics::Meter> response_enqueue_meter_;
  // std::shared_ptr<metrics::Meter> response_dequeue_meter_;
  // // std::shared_ptr<metrics::Histogram> malloc_latency_;
  // std::shared_ptr<metrics::Histogram> recv_latency_;
};

class ClientFeatureVector {
 public:
  ClientFeatureVector() = default;

  ClientFeatureVector(std::shared_ptr<void> data, size_t size_typed,
                      size_t size_bytes, DataType type);

  // Copy constructors
  ClientFeatureVector(const ClientFeatureVector &other) = default;
  ClientFeatureVector &operator=(const ClientFeatureVector &other) = default;

  // move constructors
  ClientFeatureVector(ClientFeatureVector &&other) = default;
  ClientFeatureVector &operator=(ClientFeatureVector &&other) = default;

  void *get_data();
  std::shared_ptr<void> data_;
  size_t size_typed_;
  size_t size_bytes_;
  DataType type_;
};

}  // namespace zmq_client

#endif  // CLIPPER_ZMQ_CLIENT_HPP
