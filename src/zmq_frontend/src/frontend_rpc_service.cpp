#include "frontend_rpc_service.hpp"

#include <mutex>

#include <zmq.hpp>
#include <folly/ProducerConsumerQueue.h>
#include <boost/functional/hash.hpp>

#include <clipper/config.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/redis.hpp>
#include <clipper/rpc_service.hpp>
#include <clipper/task_executor.hpp>
#include <clipper/threadpool.hpp>
#include <clipper/util.hpp>

namespace zmq_frontend {

FrontendRPCService::FrontendRPCService()
    : response_queue_(std::make_shared<folly::ProducerConsumerQueue<FrontendRPCResponse>>(RESPONSE_QUEUE_SIZE)),
      prediction_executor_(std::make_shared<wangle::CPUThreadPoolExecutor>(6)),
      active_(false) {

}

FrontendRPCService::~FrontendRPCService() {
  stop();
}

void FrontendRPCService::start(const std::string address, int port) {
  active_ = true;
  rpc_thread_ = std::thread([this, address, port]() {
    manage_service(address, port);
  });
}

void FrontendRPCService::stop() {
  if(active_) {
    active_ = false;
    rpc_thread_.join();
  }
}

void FrontendRPCService::add_application(std::string name, std::function<void(FrontendRPCRequest)> app_function) {
  std::lock_guard<std::mutex> lock(app_functions_mutex_);
  app_functions_.emplace(name, app_function);
}

void FrontendRPCService::send_response(FrontendRPCResponse response) {
  response_queue_->write(response);
}

void FrontendRPCService::manage_service(const std::string address, int port) {
  // Mapping from request id to ZMQ routing ID
  std::unordered_map<size_t, std::vector<uint8_t>> outstanding_requests_;
  size_t request_id = 0;

  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_ROUTER);
  zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
  while(active_) {
    if(response_queue_->isEmpty()) {
      if (items[0].revents & ZMQ_POLLIN) {
        receive_request(socket, request_id);
        for (int i = 0; i < NUM_REQUESTS_RECV - 1; i++) {
          zmq_poll(items, 1, 0);
          if (items[0].revents & ZMQ_POLLIN) {
            receive_request(socket, request_id);
          }
        }
      }
    } else {
      for (int i = 0; i < NUM_REQUESTS_RECV; i++) {
        zmq_poll(items, 1, 0);
        if (items[0].revents & ZMQ_POLLIN) {
          receive_request(socket, request_id);
        }
      }
    }
    send_responses(socket);
  }
}

void FrontendRPCService::receive_request(zmq::socket_t &socket,
                                         std::unordered_map<size_t, std::vector<uint8_t>>& outstanding_requests,
                                         size_t& request_id) {
  zmq::message_t msg_routing_identity;
  zmq::message_t msg_app_name;
  zmq::message_t msg_data_type;
  zmq::message_t msg_data_size_typed;

  socket.recv(&msg_routing_identity, 0);
  socket.recv(&msg_app_name, 0);
  socket.recv(&msg_data_type, 0);
  socket.recv(&msg_data_size_typed, 0);

  std::string app_name(msg_app_name.data(), msg_app_name.data() + msg_app_name.size());
  DataType input_type = static_cast<DataType>(static_cast<int*>(msg_data_type.data())[0]);
  size_t input_size_typed = static_cast<size_t*>(msg_data_size_typed.data())[0];

  std::shared_ptr<clipper::Input> input;
  switch(input_type) {
    case DataType::Bytes:
      input = std::make_shared<ByteVector>(input_size_typed);
      std::shared_ptr<uint8_t> data(static_cast<uint8_t*>(malloc(input_size_typed)), free);
      socket.recv(data.get(), input_size_typed);
      break;
    case DataType::Ints:
      input = std::make_shared<IntVector>(input_size_typed);
      std::shared_ptr<int> data(static_cast<int*>(malloc(input_size_typed * sizeof(int))), free);
      socket.recv(data.get(), input_size_typed * sizeof(int));
      break;
    case DataType::Floats:
      input = std::make_shared<FloatVector>(input_size_typed);
      std::shared_ptr<float> data(static_cast<float*>(malloc(input_size_typed * sizeof(float))), free);
      socket.recv(data.get(), input_size_typed * sizeof(float));
      break;
    case DataType::Doubles:
      input = std::make_shared<FloatVector>(input_size_typed);
      std::shared_ptr<double> data(static_cast<double*>(malloc(input_size_typed * sizeof(double))), free);
      socket.recv(data.get(), input_size_typed * sizeof(double));
      break;
    case DataType::Strings:
      input = std::make_shared<FloatVector>(input_size_typed);
      std::shared_ptr<char> data(static_cast<char*>(malloc(input_size_typed * sizeof(char))), free);
      socket.recv(data.get(), input_size_typed * sizeof(char));
      break;
    case DataType::Invalid:
    default: {
      std::stringstream ss;
      ss << "Received a request with an input with invalid type: "
         << get_readable_input_type(request_input_type);
      throw std::runtime_error(ss.str());
    }
  }

  std::lock_guard<std::mutex> lock(app_functions_);
  auto app_functions_search = app_functions_.find(app_name);
  if(app_functions_search == app_functions_.end()) {
    log_error_formatted(LOGGING_TAG_ZMQ_FRONTEND,
                        "Received a request for an unknown application with name {}",
                        app_name);
  } else {
    auto app_function = app_functions_search->second;

    int req_id = request_id;
    request_id++;

    std::vector<uint8_t> routing_id(msg_routing_identity.data(),
                                    msg_routing_identity.data() + msg_routing_identity.size());
    outstanding_requests.emplace(req_id, std::move(routing_id));

    // Submit the function call with the request to a threadpool!!!
    prediction_executor_->add([app_function, input, req_id]() {
      app_function(std::make_pair(input, req_id));
    });
  }
}

void FrontendRPCService::send_responses(zmq::socket_t &socket,
                                        std::unordered_map<size_t, std::vector<uint8_t>>& outstanding_requests) {
  size_t num_responses = NUM_RESPONSES_SEND;
  while(!response_queue_->isEmpty() && num_responses > 0) {
    FrontendRPCResponse response = response_queue_->popFront();
    auto routing_identity_search = outstanding_requests.find(response.second);
    if(routing_identity_search == outstanding_requests.end()) {
      std::stringstream ss;
      ss << "Received a response for a request with id " << response.second
         << " that has no associated routing identity";
      throw std::runtime_error(ss.str());
    }
    std::vector<uint8_t> &routing_id = routing_identity_search->second;
    int output_type = static_cast<int>(response.first.y_hat_->type());

    // TODO(czumar): If this works, include other relevant output data (default bool, default expl, etc)
    socket.send(routing_id.data(), routing_id.size(), ZMQ_SNDMORE);
    socket.send(&output_type, sizeof(int), ZMQ_SNDMORE);
    socket.send(response.first.y_hat_->get_data(), response.first.y_hat_->byte_size());
  }
}

} // namespace zmq_frontend