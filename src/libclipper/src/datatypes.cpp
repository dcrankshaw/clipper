#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include <boost/functional/hash.hpp>
#include <clipper/constants.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/util.hpp>

namespace clipper {

template <typename T>
size_t serialize_to_buffer(const std::shared_ptr<T> &data, const size_t size,
                           uint8_t *buf) {
  size_t amt_to_write = size * (sizeof(T) / sizeof(uint8_t));
  memcpy(buf, data.get(), amt_to_write);
  return amt_to_write;
}

std::string get_readable_input_type(DataType type) {
  switch (type) {
    case DataType::Bytes: return std::string("bytes");
    case DataType::Ints: return std::string("integers");
    case DataType::Floats: return std::string("floats");
    case DataType::Doubles: return std::string("doubles");
    case DataType::Strings: return std::string("strings");
    case DataType::Invalid:
    default: return std::string("Invalid input type");
  }
}

DataType parse_input_type(std::string type_string) {
  if (type_string == "bytes" || type_string == "byte" || type_string == "b") {
    return DataType::Bytes;
  } else if (type_string == "integers" || type_string == "ints" ||
             type_string == "integer" || type_string == "int" ||
             type_string == "i") {
    return DataType::Ints;
  } else if (type_string == "floats" || type_string == "float" ||
             type_string == "f") {
    return DataType::Floats;
  } else if (type_string == "doubles" || type_string == "double" ||
             type_string == "d") {
    return DataType::Doubles;
  } else if (type_string == "strings" || type_string == "string" ||
             type_string == "str" || type_string == "strs" ||
             type_string == "s") {
    return DataType::Strings;
  } else {
    throw std::invalid_argument(type_string + " is not a valid input string");
  }
}

VersionedModelId::VersionedModelId(const std::string name, const std::string id)
    : name_(name), id_(id) {}

std::string VersionedModelId::get_name() const { return name_; }

std::string VersionedModelId::get_id() const { return id_; }

std::string VersionedModelId::serialize() const {
  std::stringstream ss;
  ss << name_;
  ss << ITEM_PART_CONCATENATOR;
  ss << id_;
  return ss.str();
}

VersionedModelId VersionedModelId::deserialize(std::string str) {
  auto split = str.find(ITEM_PART_CONCATENATOR);
  std::string model_name = str.substr(0, split);
  std::string model_version = str.substr(split + 1, str.size());
  return VersionedModelId(model_name, model_version);
}

bool VersionedModelId::operator==(const VersionedModelId &rhs) const {
  return (name_ == rhs.name_ && id_ == rhs.id_);
}

bool VersionedModelId::operator!=(const VersionedModelId &rhs) const {
  return !(name_ == rhs.name_ && id_ == rhs.id_);
}

Output::Output(const std::shared_ptr<OutputData> y_hat,
               const std::vector<VersionedModelId> models_used)
    : y_hat_(std::move(y_hat)), models_used_(models_used) {}

bool Output::operator==(const Output &rhs) const {
  return (y_hat_->hash() == rhs.y_hat_->hash() &&
          models_used_ == rhs.models_used_);
}

bool Output::operator!=(const Output &rhs) const {
  return !(y_hat_->hash() == rhs.y_hat_->hash() &&
           models_used_ == rhs.models_used_);
}

ByteVector::ByteVector(std::shared_ptr<uint8_t> data, size_t size)
    : data_(data), size_(size) {}

ByteVector::ByteVector(const uint8_t* data, size_t size) : ByteVector(size) {
  set_data(data, size);
}

ByteVector::ByteVector(size_t size)
    : data_(std::shared_ptr<uint8_t>(static_cast<uint8_t*>(malloc(size)), free)), size_(size) {

}

DataType ByteVector::type() const { return DataType::Bytes; }

void ByteVector::set_data(const void *buf, size_t size) {
  memcpy(data_.get(), buf, size);
}

size_t ByteVector::serialize(uint8_t *buf) const {
  return serialize_to_buffer(data_, size_, buf);
}

size_t ByteVector::hash() const { return hash_shared_ptr(data_, size_); }

size_t ByteVector::size() const { return size_; }

size_t ByteVector::byte_size() const { return size_ * sizeof(uint8_t); }

const std::shared_ptr<uint8_t> &ByteVector::get_data() const { return data_; }

IntVector::IntVector(std::shared_ptr<int> data, size_t size)
    : data_(data), size_(size) {}

IntVector::IntVector(const int *data, size_t size) : IntVector(size) {
  set_data(data, size);
}

IntVector::IntVector(size_t size)
    : data_(std::shared_ptr<int>(static_cast<int*>(malloc(size * sizeof(int))), free)), size_(size) {

}

void IntVector::set_data(const void *buf, size_t size) {
  memcpy(data_.get(), buf, size * sizeof(int));
}

DataType IntVector::type() const { return DataType::Ints; }

size_t IntVector::serialize(uint8_t *buf) const {
  return serialize_to_buffer(data_, size_, buf);
}

size_t IntVector::hash() const { return hash_shared_ptr(data_, size_); }

size_t IntVector::size() const { return size_; }

size_t IntVector::byte_size() const { return size_ * sizeof(int); }

const std::shared_ptr<int> &IntVector::get_data() const { return data_; }

FloatVector::FloatVector(std::shared_ptr<float> data, size_t size)
    : data_(data), size_(size) {}

FloatVector::FloatVector(const float *data, size_t size) : FloatVector(size) {
  set_data(data, size);
}

FloatVector::FloatVector(size_t size)
    : data_(std::shared_ptr<float>(static_cast<float*>(malloc(size * sizeof(float))), free)), size_(size) {

}

void FloatVector::set_data(const void *buf, size_t size) {
  memcpy(data_.get(), buf, size * sizeof(float));
}

size_t FloatVector::serialize(uint8_t *buf) const {
  return serialize_to_buffer(data_, size_, buf);
}

DataType FloatVector::type() const { return DataType::Floats; }

size_t FloatVector::hash() const {
  // TODO [CLIPPER-63]: Find an alternative to hashing floats directly, as this
  // is generally a bad idea due to loss of precision from floating point
  // representations
  return hash_shared_ptr(data_, size_);
}

size_t FloatVector::size() const { return size_; }

size_t FloatVector::byte_size() const { return size_ * sizeof(float); }

const std::shared_ptr<float> &FloatVector::get_data() const { return data_; }

DoubleVector::DoubleVector(std::shared_ptr<double> data, size_t size)
    : data_(data), size_(size) {}

DoubleVector::DoubleVector(const double *data, size_t size) : DoubleVector(size) {
  set_data(data, size);
}

DoubleVector::DoubleVector(size_t size)
    : data_(std::shared_ptr<double>(static_cast<double*>(malloc(size * sizeof(double))), free)), size_(size) {

}

DataType DoubleVector::type() const { return DataType::Doubles; }

void DoubleVector::set_data(const void *buf, size_t size) {
  memcpy(data_.get(), buf, size * sizeof(double));
}

size_t DoubleVector::serialize(uint8_t *buf) const {
  return serialize_to_buffer(data_, size_, buf);
}

size_t DoubleVector::hash() const {
  // TODO [CLIPPER-63]: Find an alternative to hashing doubles directly, as
  // this is generally a bad idea due to loss of precision from floating point
  // representations
  return hash_shared_ptr(data_, size_);
}

size_t DoubleVector::size() const { return size_; }

size_t DoubleVector::byte_size() const { return size_ * sizeof(double); }

const std::shared_ptr<double> &DoubleVector::get_data() const { return data_; }

SerializableString::SerializableString(std::shared_ptr<char> data, size_t size)
    : data_(data), size_(size) {}

SerializableString::SerializableString(const char *data, size_t size) : SerializableString(size) {
  set_data(data, size);
}

SerializableString::SerializableString(size_t size)
    : data_(std::shared_ptr<char>(static_cast<char*>(malloc(size * sizeof(char))), free)), size_(size) {

}

DataType SerializableString::type() const { return DataType::Strings; }

void SerializableString::set_data(const void *buf, size_t size) {
  memcpy(data_.get(), buf, size * sizeof(char));
}

size_t SerializableString::serialize(uint8_t *buf) const {
  size_t amt_written = serialize_to_buffer(data_, size_, buf);
  buf[amt_written] = '\0';
  return amt_written + 1;
}

size_t SerializableString::hash() const {
  return hash_shared_ptr(data_, size_);
}

size_t SerializableString::size() const { return 1; }

size_t SerializableString::byte_size() const {
  // The length of the string with an extra byte for the null terminator
  return size_ + 1;
}

const std::shared_ptr<char> &SerializableString::get_data() const {
  return data_;
}

std::shared_ptr<OutputData> OutputData::create_output(
    DataType type, std::shared_ptr<void> data, size_t start, size_t end) {
  switch (type) {
    case DataType::Bytes:
      return std::make_shared<ByteVectorOutput>(
          std::static_pointer_cast<uint8_t>(data), start, end);
    case DataType::Ints:
      return std::make_shared<IntVectorOutput>(
          std::static_pointer_cast<int>(data), start / sizeof(int),
          end / sizeof(int));
    case DataType::Floats:
      return std::make_shared<FloatVectorOutput>(
          std::static_pointer_cast<float>(data), start / sizeof(float),
          end / sizeof(float));
    case DataType::Strings:
      return std::make_shared<StringOutput>(
          std::static_pointer_cast<char>(data), start / sizeof(char),
          end / sizeof(char));
    case DataType::Doubles:
    case DataType::Invalid:
    default:
      std::stringstream ss;
      ss << "Attempted to create an output of an unsupported data type: "
         << get_readable_input_type(type);
      throw std::runtime_error(ss.str());
  }
}

ByteVectorOutput::ByteVectorOutput(std::shared_ptr<uint8_t> data, size_t start,
                                   size_t end)
    : data_(data), start_(start), end_(end) {}

size_t ByteVectorOutput::size() const { return end_ - start_; }

size_t ByteVectorOutput::byte_size() const { return end_ - start_; }

size_t ByteVectorOutput::hash() const {
  return boost::hash_range(data_.get() + start_, data_.get() + end_);
}

DataType ByteVectorOutput::type() const { return DataType::Bytes; }

size_t ByteVectorOutput::serialize(void *buf) const {
  memcpy(buf, data_.get() + start_, end_ - start_);
  return end_ - start_;
}

const void* ByteVectorOutput::get_data() const {
  return data_.get() + start_;
}

IntVectorOutput::IntVectorOutput(std::shared_ptr<int> data, size_t start,
                                 size_t end)
    : data_(data), start_(start), end_(end) {}

size_t IntVectorOutput::size() const { return end_ - start_; }

size_t IntVectorOutput::byte_size() const {
  return (end_ - start_) * sizeof(int);
}

size_t IntVectorOutput::hash() const {
  return boost::hash_range(data_.get() + start_, data_.get() + end_);
}

DataType IntVectorOutput::type() const { return DataType::Ints; }

size_t IntVectorOutput::serialize(void *buf) const {
  memcpy(buf, data_.get() + start_, (end_ - start_) * sizeof(int));
  return end_ - start_;
}

const void* IntVectorOutput::get_data() const {
  return data_.get() + start_;
}

FloatVectorOutput::FloatVectorOutput(std::shared_ptr<float> data, size_t start,
                                     size_t end)
    : data_(data), start_(start), end_(end) {}

size_t FloatVectorOutput::size() const { return end_ - start_; }

size_t FloatVectorOutput::byte_size() const {
  return (end_ - start_) * sizeof(float);
}

size_t FloatVectorOutput::hash() const {
  return boost::hash_range(data_.get() + start_, data_.get() + end_);
}

DataType FloatVectorOutput::type() const { return DataType::Floats; }

size_t FloatVectorOutput::serialize(void *buf) const {
  memcpy(buf, data_.get() + start_, (end_ - start_) * sizeof(float));
  return end_ - start_;
}

const void* FloatVectorOutput::get_data() const {
  return data_.get() + start_;
}

StringOutput::StringOutput(std::shared_ptr<char> data, size_t start, size_t end)
    : data_(data), start_(start), end_(end) {}

size_t StringOutput::size() const { return end_ - start_; }

size_t StringOutput::byte_size() const {
  return (end_ - start_) * sizeof(char);
}

size_t StringOutput::hash() const {
  return boost::hash_range(data_.get() + start_, data_.get() + end_);
}

DataType StringOutput::type() const { return DataType::Strings; }

size_t StringOutput::serialize(void *buf) const {
  memcpy(buf, data_.get() + start_, (end_ - start_) * sizeof(char));
  return end_ - start_;
}

const void* StringOutput::get_data() const {
  return data_.get() + start_;
}

rpc::PredictionRequest::PredictionRequest(DataType input_type)
    : input_type_(input_type) {}

rpc::PredictionRequest::PredictionRequest(
    std::vector<std::shared_ptr<Input>> inputs, DataType input_type)
    : inputs_(inputs), input_type_(input_type) {
  for (int i = 0; i < (int)inputs.size(); i++) {
    validate_input_type(inputs[i]);
    input_data_size_ += inputs[i]->byte_size();
  }
}

void rpc::PredictionRequest::validate_input_type(
    std::shared_ptr<Input> &input) const {
  if (input->type() != input_type_) {
    std::ostringstream ss;
    ss << "Attempted to add an input of type "
       << get_readable_input_type(input->type())
       << " to a prediction request with input type "
       << get_readable_input_type(input_type_);
    log_error(LOGGING_TAG_CLIPPER, ss.str());
    throw std::invalid_argument(ss.str());
  }
}

void rpc::PredictionRequest::add_input(std::shared_ptr<Input> input) {
  validate_input_type(input);
  inputs_.push_back(input);
  input_data_size_ += input->byte_size();
}

std::vector<ByteBuffer> rpc::PredictionRequest::serialize() {
  if (input_data_size_ <= 0) {
    throw std::length_error(
        "Attempted to serialize a request with no input data!");
  }

  size_t request_metadata_size = 1 * sizeof(uint32_t);
  std::shared_ptr<uint8_t> request_metadata(
      static_cast<uint8_t *>(malloc(request_metadata_size)), free);
  uint32_t *request_metadata_raw =
      reinterpret_cast<uint32_t *>(request_metadata.get());
  request_metadata_raw[0] = static_cast<uint32_t>(RequestType::PredictRequest);

  size_t input_metadata_size = (2 + (inputs_.size() - 1)) * sizeof(uint32_t);
  std::shared_ptr<uint8_t> input_metadata(
      static_cast<uint8_t *>(malloc(input_metadata_size)), free);
  uint32_t *input_metadata_raw =
      reinterpret_cast<uint32_t *>(input_metadata.get());
  input_metadata_raw[0] = static_cast<uint32_t>(input_type_);
  input_metadata_raw[1] = static_cast<uint32_t>(inputs_.size());

  std::shared_ptr<uint8_t> input_buf(
      static_cast<uint8_t *>(malloc(input_data_size_)), free);
  uint8_t *input_buf_raw = input_buf.get();
  uint32_t index = 0;
  for (size_t i = 0; i < inputs_.size() - 1; i++) {
    size_t amt_written = inputs_[i]->serialize(input_buf_raw);
    input_buf_raw += amt_written;
    index += inputs_[i]->size();
    input_metadata_raw[i + 2] = index;
  }
  // Don't include the final separation index because it results in the
  // creation of an empty data array when deserializing
  size_t tail_index = inputs_.size() - 1;
  size_t amt_written = inputs_[tail_index]->serialize(input_buf_raw);
  input_buf_raw += amt_written;

  size_t input_metadata_size_buf_size = 1 * sizeof(long);
  std::shared_ptr<uint8_t> input_metadata_size_buf(
      static_cast<uint8_t *>(malloc(input_metadata_size_buf_size)), free);
  long *input_metadata_size_buf_raw =
      reinterpret_cast<long *>(input_metadata_size_buf.get());
  // Add the size of the input metadata in bytes. This will be
  // sent prior to the input metadata to allow for proactive
  // buffer allocation in the receiving container
  input_metadata_size_buf_raw[0] = input_metadata_size;

  size_t inputs_size_buf_size = 1 * sizeof(long);
  std::shared_ptr<uint8_t> inputs_size_buf(
      static_cast<uint8_t *>(malloc(inputs_size_buf_size)), free);
  long *inputs_size_buf_raw = reinterpret_cast<long *>(inputs_size_buf.get());
  // Add the size of the serialized inputs in bytes. This will be
  // sent prior to the input data to allow for proactive
  // buffer allocation in the receiving container
  inputs_size_buf_raw[0] = input_data_size_;

  std::vector<ByteBuffer> serialized_request;
  serialized_request.push_back(
      std::make_pair(request_metadata, request_metadata_size));
  serialized_request.push_back(
      std::make_pair(input_metadata_size_buf, input_metadata_size_buf_size));
  serialized_request.push_back(
      std::make_pair(input_metadata, input_metadata_size));
  serialized_request.push_back(
      std::make_pair(inputs_size_buf, inputs_size_buf_size));
  serialized_request.push_back(std::make_pair(input_buf, input_data_size_));

  return serialized_request;
}

rpc::PredictionResponse::PredictionResponse(
    const std::vector<std::shared_ptr<OutputData>> outputs)
    : outputs_(outputs) {}

rpc::PredictionResponse
rpc::PredictionResponse::deserialize_prediction_response(
    DataType data_type, std::shared_ptr<void> &data) {
  std::vector<std::shared_ptr<OutputData>> outputs;
  uint32_t *output_lengths_data = reinterpret_cast<uint32_t *>(data.get());
  uint32_t num_outputs = output_lengths_data[0];
  output_lengths_data++;
  size_t curr_output_index =
      sizeof(uint32_t) + (num_outputs * sizeof(uint32_t));
  for (uint32_t i = 0; i < num_outputs; i++) {
    uint32_t output_length = output_lengths_data[i];
    std::shared_ptr<OutputData> output = OutputData::create_output(
        data_type, data, curr_output_index, curr_output_index + output_length);
    outputs.push_back(std::move(output));
    curr_output_index += output_length;
  }
  return PredictionResponse(outputs);
}

Query::Query(std::string label, long user_id, std::shared_ptr<Input> input,
             long latency_budget_micros, std::string selection_policy,
             std::vector<VersionedModelId> candidate_models)
    : label_(label),
      user_id_(user_id),
      input_(input),
      latency_budget_micros_(latency_budget_micros),
      selection_policy_(selection_policy),
      candidate_models_(candidate_models),
      create_time_(std::chrono::high_resolution_clock::now()) {}

Response::Response(Query query, QueryId query_id, const long duration_micros,
                   Output output, const bool output_is_default,
                   const boost::optional<std::string> default_explanation)
    : query_(std::move(query)),
      query_id_(query_id),
      duration_micros_(duration_micros),
      output_(std::move(output)),
      output_is_default_(output_is_default),
      default_explanation_(std::move(default_explanation)) {}

Feedback::Feedback(std::shared_ptr<Input> input, double y)
    : y_(y), input_(input) {}

FeedbackQuery::FeedbackQuery(std::string label, long user_id, Feedback feedback,
                             std::string selection_policy,
                             std::vector<VersionedModelId> candidate_models)
    : label_(label),
      user_id_(user_id),
      feedback_(feedback),
      selection_policy_(selection_policy),
      candidate_models_(candidate_models) {}

PredictTask::PredictTask(std::shared_ptr<Input> input, VersionedModelId model,
                         float utility, QueryId query_id,
                         long latency_slo_micros)
    : input_(std::move(input)),
      model_(model),
      utility_(utility),
      query_id_(query_id),
      latency_slo_micros_(latency_slo_micros) {}

FeedbackTask::FeedbackTask(Feedback feedback, VersionedModelId model,
                           QueryId query_id, long latency_slo_micros)
    : feedback_(feedback),
      model_(model),
      query_id_(query_id),
      latency_slo_micros_(latency_slo_micros) {}

}  // namespace clipper
