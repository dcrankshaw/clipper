#include <random>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>

#include <boost/optional.hpp>

#include "zmq_client.hpp"

using namespace clipper;
using namespace zmq_client;

static const std::string RES50 = "res50";
static const std::string RES152 = "res152";
static const std::string ALEXNET = "alexnet";
static const std::string TF_KERNEL_SVM = "tf-kernel-svm";
static const std::string INCEPTION_FEATS = "inception";
static const std::string TF_LOG_REG = "tf-log-reg";
static const std::string TF_RESNET = "tf-resnet-feats";
static const std::string TF_RESNET_VAR = "tf-resnet-feats-var";
static const std::string TF_RESNET_SLEEP = "tf-resnet-feats-sleep";

static const std::string TF_LANG_DETECT = "tf-lang-detect";
static const std::string TF_NMT = "tf-nmt";
static const std::string TF_LSTM = "tf-lstm";

const std::vector<std::string> FLOAT_VECTOR_MODELS{RES50,         RES152,          ALEXNET,
                                                   TF_KERNEL_SVM, INCEPTION_FEATS, TF_LOG_REG,
                                                   TF_RESNET,     TF_RESNET_VAR,   TF_RESNET_SLEEP};

std::vector<ClientFeatureVector> generate_text_inputs(const std::string& workload_path,
                                                      size_t desired_input_length) {
  int num_inputs = 1000;
  std::random_device rd;         // Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  std::vector<ClientFeatureVector> all_inputs;
  std::ifstream text_file(workload_path);

  std::string line;
  while (std::getline(text_file, line)) {
    size_t input_size_bytes = line.size() * sizeof(char);
    size_t desired_input_length_bytes = desired_input_length * sizeof(char);
    size_t cp_unit_size = std::min(input_size_bytes, desired_input_length_bytes);

    std::shared_ptr<void> input_data(malloc(desired_input_length_bytes), free);
    char* raw_input_data = static_cast<char*>(input_data.get());
    size_t curr_cp_idx = 0;
    size_t curr_size = 0;
    while (curr_size < desired_input_length) {
      size_t curr_unit_size = std::min(cp_unit_size, desired_input_length - curr_size);
      memcpy(raw_input_data + curr_cp_idx, line.data(), curr_unit_size);
      curr_size += curr_unit_size;
      curr_cp_idx += curr_unit_size;
    }
    ClientFeatureVector input(input_data, desired_input_length, desired_input_length_bytes,
                              DataType::Bytes);
    all_inputs.push_back(std::move(input));
  }

  std::vector<ClientFeatureVector> selected_inputs;
  selected_inputs.reserve(num_inputs);

  for (size_t i = 0; i < num_inputs; i++) {
    size_t idx = static_cast<size_t>(distribution(generator) * all_inputs.size());
    selected_inputs.push_back(all_inputs[idx]);
  }
  return selected_inputs;
}

std::vector<ClientFeatureVector> generate_float_inputs(int input_length) {
  int num_points = 1000;
  std::random_device rd;         // Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distribution(0.0, 1.0);
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

std::vector<ClientFeatureVector> generate_inputs(const std::string& model_name, size_t input_size, boost::optional<std::string> workload_path = boost::optional<std::string>()) {
  if (std::find(FLOAT_VECTOR_MODELS.begin(), FLOAT_VECTOR_MODELS.end(), model_name) !=
      FLOAT_VECTOR_MODELS.end()) {
    return generate_float_inputs(input_size);
  } else if (model_name == TF_LANG_DETECT) {
    return generate_text_inputs(workload_path.get(), input_size);
  } else if (model_name == TF_LSTM) {
    return generate_text_inputs(workload_path.get(), input_size);
  } else if (model_name == TF_NMT) {
    return generate_text_inputs(workload_path.get(), input_size);
  } else {
    std::stringstream ss;
    ss << "Attempted to generate inputs for unsupported model: " << model_name;
    throw std::runtime_error(ss.str());
  }
}
