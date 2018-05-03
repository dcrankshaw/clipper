#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <cxxopts.hpp>

#include <clipper/clock.hpp>
#include <clipper/metrics.hpp>

#include "client_metrics.hpp"
#include "driver.hpp"
#include "zmq_client.hpp"

using namespace clipper;
using namespace zmq_client;

static const std::string INCEPTION_FEATS = "tf-inception-contention";
// static const std::string TF_KERNEL_SVM = "tf-kernel-svm-contention";
static const std::string CASCADE_PREPROCESS = "cascadepreprocess-contention";

void predict(std::unordered_map<std::string, std::shared_ptr<FrontendRPCClient>> clients,
    ClientFeatureVector inception_input, std::atomic<int>& prediction_counter) {
  // size_t ksvm_input_length = 2048;
  // ClientFeatureVector ksvm_input(inception_input.data_, ksvm_input_length,
  //                                  ksvm_input_length * sizeof(float), DataType::Floats);

  std::shared_ptr<std::atomic_int> branches_completed = std::make_shared<std::atomic_int>(0);
  // NOOP
  auto callback = [branches_completed, &prediction_counter](
      ClientFeatureVector output, std::shared_ptr<QueryLineage> lineage) {
    int num_branches_completed = branches_completed->fetch_add(1);
    if (num_branches_completed == 1) {
      prediction_counter += 1;
    }
  };

  // clients[TF_KERNEL_SVM]->send_request(TF_KERNEL_SVM, ksvm_input, callback);
  clients[CASCADE_PREPROCESS]->send_request(CASCADE_PREPROCESS, ksvm_input, callback);
  clients[INCEPTION_FEATS]->send_request(INCEPTION_FEATS, inception_input, callback);
}

std::vector<ClientFeatureVector> generate_float_inputs(int input_length) {
  int num_points = 1000;
  std::random_device rd;         // Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  std::vector<ClientFeatureVector> inputs;
  for (int i = 0; i < num_points; ++i) {
    float* input_buffer = reinterpret_cast<float*>(malloc(input_length * sizeof(float)));
    for (int j = 0; j < input_length; ++j) {
      input_buffer[j] = distribution(generator);
    }
    std::shared_ptr<void> input_ptr(reinterpret_cast<void*>(input_buffer), free);
    inputs.emplace_back(input_ptr, input_length, input_length * sizeof(float), DataType::Floats);
  }
  return inputs;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("image_driver_one", "Image Driver One");
  // clang-format off
  options.add_options()
      ("target_throughput", "Mean throughput to target in qps",
       cxxopts::value<float>())
      // ("request_distribution", "Distribution to sample request delay from. "
      //  "Can be 'constant', 'poisson', or 'batch', or 'file'. 'batch' sends a single batch at a time."
      //  "'file' uses the delays provided in the request_delay_file argument.",
      //  cxxopts::value<std::string>())
      // ("trial_length", "Number of queries per trial",
      //  cxxopts::value<int>())
      // ("num_trials", "Number of trials",
      //  cxxopts::value<int>())
      // ("log_file", "location of log file",
      //  cxxopts::value<std::string>())
      ("clipper_address", "IP address or hostname of ZMQ frontend",
       cxxopts::value<std::string>())
      // ("clipper_address_inception", "IP address or hostname of ZMQ frontend to user for the inception branch",
      //  cxxopts::value<std::string>())
      // ("request_delay_file", "Path to file containing a list of inter-arrival delays, one per line.",
      //  cxxopts::value<std::string>())
      // ("get_clipper_metrics", "Collect Clipper metrics",
      //  cxxopts::value<bool>())
       ;
  // clang-format on
  options.parse(argc, argv);
  std::string distribution = "constant";
  std::string log_file = "/tmp/contention-driver";

  // Request the system uptime so that a clock instance is created as
  // soon as the frontend starts
  std::vector<ClientFeatureVector> inputs = generate_float_inputs(299 * 299 * 3);
  std::vector<std::string> models = {INCEPTION_FEATS, TF_KERNEL_SVM};


  auto predict_func = [](
      std::unordered_map<std::string, std::shared_ptr<FrontendRPCClient>> clients, ClientFeatureVector input,
      std::atomic<int>& prediction_counter) {
    predict(clients, input, prediction_counter);
  };
  std::unordered_map<std::string, std::string> addresses;
  addresses.emplace(INCEPTION_FEATS, options["clipper_address"].as<std::string>());
  addresses.emplace(CASCADE_PREPROCESS, options["clipper_address"].as<std::string>());
  Driver driver(predict_func, std::move(inputs), options["target_throughput"].as<float>(),
                distribution, 100000, 100000, /* Basically just run forever */
                log_file, addresses, -1, {},
                false);
  std::cout << "Starting CONTENTION driver" << std::endl;

  driver.start();
  std::cout << "Driver completed" << std::endl;
  return 0;
}
