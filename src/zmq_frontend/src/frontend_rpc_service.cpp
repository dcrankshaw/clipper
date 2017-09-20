#include "frontend_rpc_service.hpp"

#include <mutex>

#include <folly/ProducerConsumerQueue.h>
#include <boost/functional/hash.hpp>
#include <zmq.hpp>

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
    : response_queue_(
          std::make_shared<folly::ProducerConsumerQueue<FrontendRPCResponse>>(
              RESPONSE_QUEUE_SIZE)),
      prediction_executor_(std::make_shared<wangle::CPUThreadPoolExecutor>(6)),
      active_(false) {}

FrontendRPCService::~FrontendRPCService() { stop(); }

void FrontendRPCService::start(const std::string address, int send_port, int recv_port) {
  active_ = true;
  rpc_send_thread_ = std::thread([this, address, send_port]() {
    manage_send_service(address, send_port);
  });
  rpc_recv_thread_ = std::thread([this, address, recv_port]() {
    manage_recv_service(address, recv_port);
  });
}

void FrontendRPCService::stop() {
  if (active_) {
    active_ = false;
    rpc_send_thread_.join();
    rpc_recv_thread_.join();
  }
}

void FrontendRPCService::add_application(
    std::string name, std::function<void(FrontendRPCRequest)> app_function) {
  std::lock_guard<std::mutex> lock(app_functions_mutex_);
  app_functions_.emplace(name, app_function);
}

void FrontendRPCService::send_response(FrontendRPCResponse response) {
  std::lock_guard<std::mutex> lock(response_queue_insertion_mutex_);
  response_queue_->write(response);
}

void FrontendRPCService::manage_send_service(const std::string ip, int port) {
  std::string send_address = "tcp://" + ip + ":" + std::to_string(port);
  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_ROUTER);
  socket.bind(send_address);
  zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
  int client_id = 0;
  while(active_) {
    zmq_poll(items, 1, 1);
    if (items[0].revents & ZMQ_POLLIN) {
      handle_new_connection(socket, client_id);
    }
    send_responses(socket, NUM_RESPONSES_SEND);
  }
  shutdown_service(socket);
}

void FrontendRPCService::manage_recv_service(const std::string ip, int port) {
  std::string recv_address = "tcp://" + ip + ":" + std::to_string(port);
  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_ROUTER);
  socket.bind(recv_address);
  zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
  int request_id = 0;
  while(active_) {
    zmq_poll(items, 1, 1);
    if (items[0].revents & ZMQ_POLLIN) {
      receive_request(socket, request_id);
    }
  }
  shutdown_service(socket);
}

void FrontendRPCService::handle_new_connection(zmq::socket_t &socket, int &client_id) {
  zmq::message_t msg_routing_identity;
  zmq::message_t msg_delimiter;
  zmq::message_t msg_establish_connection;

  socket.recv(&msg_routing_identity, 0);
  socket.recv(&msg_delimiter, 0);
  socket.recv(&msg_establish_connection, 0);

  const vector<uint8_t> routing_id(
      (uint8_t *)msg_routing_identity.data(),
      (uint8_t *)msg_routing_identity.data() + msg_routing_identity.size());
  int curr_client_id = client_id++;
  std::unique_lock<std::mutex> lock(client_routing_mutex_);
  client_routing_map_.emplace(curr_client_id, std::move(routing_id));

  zmq::message_t msg_client_id(sizeof(int));
  memcpy(msg_client_id.data(), &client_id, sizeof(int));
  socket.send(msg_routing_identity, ZMQ_SNDMORE);
  socket.send("", 0, ZMQ_SNDMORE);
  socket.send(msg_client_id, 0);
}

void FrontendRPCService::shutdown_service(zmq::socket_t &socket) {
  size_t buf_size = 32;
  std::vector<char> buf(buf_size);
  socket.getsockopt(ZMQ_LAST_ENDPOINT, (void *)buf.data(), &buf_size);
  std::string last_endpoint = std::string(buf.begin(), buf.end());
  socket.unbind(last_endpoint);
  socket.close();
}

void FrontendRPCService::receive_request(zmq::socket_t &socket, int &request_id) {
  zmq::message_t msg_routing_identity;
  zmq::message_t msg_delimiter;
  zmq::message_t msg_client_id;
  zmq::message_t msg_app_name;
  zmq::message_t msg_data_type;
  zmq::message_t msg_data_size_typed;

  socket.recv(&msg_routing_identity, 0);
  socket.recv(&msg_delimiter, 0);
  socket.recv(&msg_client_id, 0);
  socket.recv(&msg_app_name, 0);
  socket.recv(&msg_data_type, 0);
  socket.recv(&msg_data_size_typed, 0);

  std::string app_name(static_cast<char *>(msg_app_name.data()),
                       msg_app_name.size());
  DataType input_type =
      static_cast<DataType>(static_cast<int *>(msg_data_type.data())[0]);
  int input_size_typed = static_cast<int *>(msg_data_size_typed.data())[0];

  std::shared_ptr<clipper::Input> input;
  switch (input_type) {
    case DataType::Bytes: {
      std::shared_ptr<uint8_t> data(
          static_cast<uint8_t *>(malloc(input_size_typed)), free);
      socket.recv(data.get(), input_size_typed, 0);
      input = std::make_shared<ByteVector>(data, input_size_typed);
    } break;
    case DataType::Ints: {
      std::shared_ptr<int> data(
          static_cast<int *>(malloc(input_size_typed * sizeof(int))), free);
      socket.recv(data.get(), input_size_typed * sizeof(int), 0);
      input = std::make_shared<IntVector>(data, input_size_typed);
    } break;
    case DataType::Floats: {
      std::shared_ptr<float> data(
          static_cast<float *>(malloc(input_size_typed * sizeof(float))), free);
      socket.recv(data.get(), input_size_typed * sizeof(float), 0);
      input = std::make_shared<FloatVector>(data, input_size_typed);
    } break;
    case DataType::Doubles: {
      std::shared_ptr<double> data(
          static_cast<double *>(malloc(input_size_typed * sizeof(double))),
          free);
      socket.recv(data.get(), input_size_typed * sizeof(double), 0);
      input = std::make_shared<DoubleVector>(data, input_size_typed);
    } break;
    case DataType::Strings: {
      std::shared_ptr<char> data(
          static_cast<char *>(malloc(input_size_typed * sizeof(char))), free);
      socket.recv(data.get(), input_size_typed * sizeof(char), 0);
      input = std::make_shared<SerializableString>(data, input_size_typed);
    } break;
    case DataType::Invalid:
    default: {
      std::stringstream ss;
      ss << "Received a request with an input with invalid type: "
         << get_readable_input_type(input_type);
      throw std::runtime_error(ss.str());
    }
  }

  std::lock_guard<std::mutex> lock(app_functions_mutex_);
  auto app_functions_search = app_functions_.find(app_name);
  if (app_functions_search == app_functions_.end()) {
    log_error_formatted(
        LOGGING_TAG_ZMQ_FRONTEND,
        "Received a request for an unknown application with name {}", app_name);
  } else {
    auto app_function = app_functions_search->second;

    int req_id = request_id;
    request_id++;

    int client_id = static_cast<int *>(msg_client_id.data())[0];

    // Submit the function call with the request to a threadpool!!!
    prediction_executor_->add([app_function, input, req_id, client_id]() {
      app_function(std::make_tuple(input, req_id, client_id));
    });
  }
}

void FrontendRPCService::send_responses(zmq::socket_t &socket, size_t num_responses) {
  while (!response_queue_->isEmpty() && num_responses > 0) {
    FrontendRPCResponse *response = response_queue_->frontPtr();
    Output &output = std::get<0>(*response);
    int request_id = std::get<1>(*response);
    int client_id = std::get<2>(*response);

    std::unique_lock<std::mutex> routing_lock(client_routing_mutex_);
    auto routing_id_search = client_routing_map_.find(client_id);
    if (routing_id_search == client_routing_map_.end()) {
      std::stringstream ss;
      ss << "Received a response associated with a client id " << client_id
         << " that has no associated routing identity";
      throw std::runtime_error(ss.str());
    }

    const std::vector<uint8_t>& routing_id = routing_id_search->second;
    routing_lock.release();

    int output_type = static_cast<int>(output.y_hat_->type());

    // TODO(czumar): If this works, include other relevant output data (default
    // bool, default expl, etc)
    socket.send(routing_id.data(), routing_id.size(), ZMQ_SNDMORE);
    socket.send("", 0, ZMQ_SNDMORE);
    socket.send(&output_type, sizeof(int), ZMQ_SNDMORE);
    socket.send(output.y_hat_->get_data(),
                output.y_hat_->byte_size());

    // Remove the response from the outbound queue now that we're done
    // processing it
    response_queue_->popFront();
    // Remove the oustanding request from the map

    num_responses--;
  }
}

}  // namespace zmq_frontend
