#include <concurrentqueue.h>

#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/threadpool.hpp>
#include <clipper/util.hpp>

using zmq::socket_t;
using zmq::message_t;
using zmq::context_t;
using std::shared_ptr;
using std::string;
using std::vector;

namespace zmq_client {

void FrontendRPCClient::manage_send_service(const std::string ip, int port) {
  const string send_address = "tcp://" + ip + ":" + std::to_string(port);
  context_t context = context_t(1);
  socket_t socket = socket_t(context, ZMQ_DEALER);
  socket.connect(send_address)
  int num_send = -1;
  while (active_) {
    if (client_id_ == -1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    else {
      send_messages(socket, num_send);
    }
  }
  socket.disconnect(send_address)
}

void FrontendRPCClient::send_messages(socket_t &socket, int max_num_messages) {
  if (max_num_messages == -1) {
    max_num_messages = request_queue_->size_approx();
  }
  InputVector input;
  size_t sent_requests = 0;
  while (sent_requests < max_num_messages && request_queue_->try_queue(input)) {
    socket.send("", ZMQ_SNDMORE);
    socket.send("", ZMQ_SNDMORE);

  }


}


}
