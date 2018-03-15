#ifndef CLIPPER_PREDICTOR_HPP
#define CLIPPER_PREDICTOR_HPP

#include <clipper/metrics.hpp>
#include "zmq_client.hpp"

namespace zmq_client {

using namespace clipper;

void spin_sleep(int duration_micros);

class Driver {
 public:
  Driver(std::function<void(FrontendRPCClient&, ClientFeatureVector,
                            std::atomic<int>&)>
             predict_func,
         std::vector<ClientFeatureVector> inputs, float target_throughput,
         std::string distribution, int trial_length, int num_trials,
         std::string log_file, std::string clipper_address);

  void start();

 private:
  void monitor_results();
  std::function<void(FrontendRPCClient&, ClientFeatureVector,
                     std::atomic<int>&)>
      predict_func_;
  std::vector<ClientFeatureVector> inputs_;
  float target_throughput_;
  std::string distribution_;
  int trial_length_;
  int num_trials_;
  std::string log_file_;
  FrontendRPCClient client_;
  std::atomic_bool done_;
  std::atomic<int> prediction_counter_;
  std::string clipper_address_;
};

}  // namespace zmq_client

#endif  // CLIPPER_PREDICTOR_HPP
