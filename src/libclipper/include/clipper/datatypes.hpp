#ifndef CLIPPER_LIB_DATATYPES_H
#define CLIPPER_LIB_DATATYPES_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>
#include <boost/thread.hpp>

namespace clipper {

typedef std::pair<std::shared_ptr<uint8_t>, size_t> ByteBuffer;

using QueryId = long;
using FeedbackAck = bool;

enum class DataType {
  Invalid = -1,
  Bytes = 0,
  Ints = 1,
  Floats = 2,
  Doubles = 3,
  Strings = 4,
};

enum class RequestType {
  PredictRequest = 0,
  FeedbackRequest = 1,
};

std::string get_readable_input_type(DataType type);
DataType parse_input_type(std::string type_string);

class VersionedModelId {
 public:
  VersionedModelId(const std::string name, const std::string id);

  std::string get_name() const;
  std::string get_id() const;
  std::string serialize() const;
  static VersionedModelId deserialize(std::string);

  VersionedModelId(const VersionedModelId &) = default;
  VersionedModelId &operator=(const VersionedModelId &) = default;

  VersionedModelId(VersionedModelId &&) = default;
  VersionedModelId &operator=(VersionedModelId &&) = default;

  bool operator==(const VersionedModelId &rhs) const;
  bool operator!=(const VersionedModelId &rhs) const;

 private:
  std::string name_;
  std::string id_;
};

class Input {
 public:
  // TODO: pure virtual or default?
  // virtual ~Input() = default;

  virtual DataType type() const = 0;

  /**
   * Sets the content of the input to the content
   * of the provided data buffer. This does
   * not take ownership of the provided buffer.
   *
   * @param buf Buffer containing input data
   * @param size The size, in elements, of the data buffer
   */
  virtual void set_data(const void* buf, size_t size) = 0;

  /**
   * Serializes input and writes resulting data to provided buffer.
   *
   * The serialization methods are used for RPC.
   */
  virtual size_t serialize(uint8_t *buf) const = 0;

  virtual size_t hash() const = 0;

  /**
   * @return The number of elements in the input
   */
  virtual size_t size() const = 0;
  /**
   * @return The size of the input data in bytes
   */
  virtual size_t byte_size() const = 0;
};

class ByteVector : public Input {
 public:
  explicit ByteVector(std::shared_ptr<uint8_t> data, size_t size);
  explicit ByteVector(const uint8_t* data, size_t size);
  explicit ByteVector(size_t size);

  // Disallow copy
  ByteVector(ByteVector &other) = delete;
  ByteVector &operator=(ByteVector &other) = delete;

  // move constructors
  ByteVector(ByteVector &&other) = default;
  ByteVector &operator=(ByteVector &&other) = default;

  DataType type() const override;
  void set_data(const void* buf, size_t size) override;
  size_t serialize(uint8_t *buf) const override;
  size_t hash() const override;
  size_t size() const override;
  size_t byte_size() const override;
  const std::shared_ptr<uint8_t> &get_data() const;

 private:
  const std::shared_ptr<uint8_t> data_;
  const size_t size_;
};

class IntVector : public Input {
 public:
  explicit IntVector(std::shared_ptr<int> data, size_t size);
  explicit IntVector(const int* data, size_t size);
  explicit IntVector(size_t size);

  // Disallow copy
  IntVector(IntVector &other) = delete;
  IntVector &operator=(IntVector &other) = delete;

  // move constructors
  IntVector(IntVector &&other) = default;
  IntVector &operator=(IntVector &&other) = default;

  DataType type() const override;
  void set_data(const void* buf, size_t size) override;
  size_t serialize(uint8_t *buf) const override;
  size_t hash() const override;
  size_t size() const override;
  size_t byte_size() const override;

  const std::shared_ptr<int> &get_data() const;

 private:
  const std::shared_ptr<int> data_;
  const size_t size_;
};

class FloatVector : public Input {
 public:
  explicit FloatVector(std::shared_ptr<float> data, size_t size);
  explicit FloatVector(const float* data, size_t size);
  explicit FloatVector(size_t size);

  // Disallow copy
  FloatVector(FloatVector &other) = delete;
  FloatVector &operator=(FloatVector &other) = delete;

  // move constructors
  FloatVector(FloatVector &&other) = default;
  FloatVector &operator=(FloatVector &&other) = default;

  DataType type() const override;
  void set_data(const void* buf, size_t size) override;
  size_t serialize(uint8_t *buf) const override;
  size_t hash() const override;
  size_t size() const override;
  size_t byte_size() const override;

  const std::shared_ptr<float> &get_data() const;

 private:
  const std::shared_ptr<float> data_;
  const size_t size_;
};

class DoubleVector : public Input {
 public:
  explicit DoubleVector(std::shared_ptr<double> data, size_t size);
  explicit DoubleVector(const double* data, size_t size);
  explicit DoubleVector(size_t size);

  // Disallow copy
  DoubleVector(DoubleVector &other) = delete;
  DoubleVector &operator=(DoubleVector &other) = delete;

  // move constructors
  DoubleVector(DoubleVector &&other) = default;
  DoubleVector &operator=(DoubleVector &&other) = default;

  DataType type() const override;
  void set_data(const void* buf, size_t size) override;
  size_t serialize(uint8_t *buf) const override;
  size_t hash() const override;
  size_t size() const override;
  size_t byte_size() const override;

  const std::shared_ptr<double> &get_data() const;

 private:
  const std::shared_ptr<double> data_;
  const size_t size_;
};

class SerializableString : public Input {
 public:
  explicit SerializableString(std::shared_ptr<char> data, size_t size);
  explicit SerializableString(const char* data, size_t size);
  explicit SerializableString(size_t size);

  // Disallow copy
  SerializableString(SerializableString &other) = delete;
  SerializableString &operator=(SerializableString &other) = delete;

  // move constructors
  SerializableString(SerializableString &&other) = default;
  SerializableString &operator=(SerializableString &&other) = default;

  DataType type() const override;
  void set_data(const void* buf, size_t size) override;
  size_t serialize(uint8_t *buf) const override;
  size_t hash() const override;
  size_t size() const override;
  size_t byte_size() const override;
  const std::shared_ptr<char> &get_data() const;

 private:
  const std::shared_ptr<char> data_;
  const size_t size_;
};

class Query {
 public:
  ~Query() = default;

  Query(std::string label, long user_id, std::shared_ptr<Input> input,
        long latency_budget_micros, std::string selection_policy,
        std::vector<VersionedModelId> candidate_models);

  // Note that it should be relatively cheap to copy queries because
  // the actual input won't be copied
  // copy constructors
  Query(const Query &) = default;
  Query &operator=(const Query &) = default;

  // move constructors
  Query(Query &&) = default;
  Query &operator=(Query &&) = default;

  // Used to provide a namespace for queries. The expected
  // use is to distinguish queries coming from different
  // REST endpoints.
  std::string label_;
  long user_id_;
  std::shared_ptr<Input> input_;
  // TODO change this to a deadline instead of a duration
  long latency_budget_micros_;
  std::string selection_policy_;
  std::vector<VersionedModelId> candidate_models_;
  std::chrono::time_point<std::chrono::high_resolution_clock> create_time_;
};

class Feedback {
 public:
  ~Feedback() = default;
  Feedback(std::shared_ptr<Input> input, double y);

  Feedback(const Feedback &) = default;
  Feedback &operator=(const Feedback &) = default;

  Feedback(Feedback &&) = default;
  Feedback &operator=(Feedback &&) = default;

  double y_;
  std::shared_ptr<Input> input_;
};

class FeedbackQuery {
 public:
  ~FeedbackQuery() = default;
  FeedbackQuery(std::string label, long user_id, Feedback feedback,
                std::string selection_policy,
                std::vector<VersionedModelId> candidate_models);

  FeedbackQuery(const FeedbackQuery &) = default;
  FeedbackQuery &operator=(const FeedbackQuery &) = default;

  FeedbackQuery(FeedbackQuery &&) = default;
  FeedbackQuery &operator=(FeedbackQuery &&) = default;

  // Used to provide a namespace for queries. The expected
  // use is to distinguish queries coming from different
  // REST endpoints.
  std::string label_;
  long user_id_;
  Feedback feedback_;
  std::string selection_policy_;
  std::vector<VersionedModelId> candidate_models_;
};

class PredictTask {
 public:
  ~PredictTask() = default;

  PredictTask(std::shared_ptr<Input> input, VersionedModelId model,
              float utility, QueryId query_id, long latency_slo_micros);

  PredictTask(const PredictTask &other) = default;

  PredictTask &operator=(const PredictTask &other) = default;
  PredictTask(PredictTask &&other) = default;

  PredictTask &operator=(PredictTask &&other) = default;

  std::shared_ptr<Input> input_;
  VersionedModelId model_;
  float utility_;
  QueryId query_id_;
  long latency_slo_micros_;
  std::chrono::time_point<std::chrono::system_clock> recv_time_;
};

/// NOTE: If a feedback task is scheduled, the task scheduler
/// must send it to ALL replicas of the VersionedModelId.
class FeedbackTask {
 public:
  ~FeedbackTask() = default;

  FeedbackTask(Feedback feedback, VersionedModelId model, QueryId query_id,
               long latency_slo_micros);

  FeedbackTask(const FeedbackTask &other) = default;

  FeedbackTask &operator=(const FeedbackTask &other) = default;

  FeedbackTask(FeedbackTask &&other) = default;

  FeedbackTask &operator=(FeedbackTask &&other) = default;

  Feedback feedback_;
  VersionedModelId model_;
  QueryId query_id_;
  long latency_slo_micros_;
};

class OutputData {
 public:
  virtual DataType type() const = 0;

  /**
   * Serializes input and writes resulting data to provided buffer.
   *
   * The serialization methods are used for RPC.
   */
  virtual size_t serialize(void *buf) const = 0;

  virtual size_t hash() const = 0;

  /**
   * @return The number of elements in the output
   */
  virtual size_t size() const = 0;
  /**
   * @return The size of the output data in bytes
   */
  virtual size_t byte_size() const = 0;

  virtual const void* get_data() const = 0;

  static std::shared_ptr<OutputData> create_output(DataType type,
                                                   std::shared_ptr<void> data,
                                                   size_t start, size_t end);
};

class FloatVectorOutput : public OutputData {
 public:
  explicit FloatVectorOutput(std::shared_ptr<float> data, size_t start,
                             size_t end);

  // Disallow copy
  FloatVectorOutput(FloatVectorOutput &other) = delete;
  FloatVectorOutput &operator=(FloatVectorOutput &other) = delete;

  // move constructors
  FloatVectorOutput(FloatVectorOutput &&other) = default;
  FloatVectorOutput &operator=(FloatVectorOutput &&other) = default;

  DataType type() const override;
  size_t serialize(void *buf) const override;
  size_t hash() const override;
  size_t size() const override;
  size_t byte_size() const override;
  const void* get_data() const override;

 private:
  const std::shared_ptr<float> data_;
  const size_t start_;
  const size_t end_;
};

class IntVectorOutput : public OutputData {
 public:
  explicit IntVectorOutput(std::shared_ptr<int> data, size_t start, size_t end);

  // Disallow copy
  IntVectorOutput(IntVectorOutput &other) = delete;
  IntVectorOutput &operator=(IntVectorOutput &other) = delete;

  // move constructors
  IntVectorOutput(IntVectorOutput &&other) = default;
  IntVectorOutput &operator=(IntVectorOutput &&other) = default;

  DataType type() const override;
  size_t serialize(void *buf) const override;
  size_t hash() const override;
  size_t size() const override;
  size_t byte_size() const override;
  const void* get_data() const override;

 private:
  const std::shared_ptr<int> data_;
  const size_t start_;
  const size_t end_;
};

class ByteVectorOutput : public OutputData {
 public:
  explicit ByteVectorOutput(std::shared_ptr<uint8_t> data, size_t start,
                            size_t end);

  // Disallow copy
  ByteVectorOutput(ByteVectorOutput &other) = delete;
  ByteVectorOutput &operator=(ByteVectorOutput &other) = delete;

  // move constructors
  ByteVectorOutput(ByteVectorOutput &&other) = default;
  ByteVectorOutput &operator=(ByteVectorOutput &&other) = default;

  DataType type() const override;
  size_t serialize(void *buf) const override;
  size_t hash() const override;
  size_t size() const override;
  size_t byte_size() const override;
  const void* get_data() const override;

 private:
  const std::shared_ptr<uint8_t> data_;
  const size_t start_;
  const size_t end_;
};

class StringOutput : public OutputData {
 public:
  explicit StringOutput(std::shared_ptr<char> data, size_t start, size_t end);

  // Disallow copy
  StringOutput(StringOutput &other) = delete;
  StringOutput &operator=(StringOutput &other) = delete;

  // move constructors
  StringOutput(StringOutput &&other) = default;
  StringOutput &operator=(StringOutput &&other) = default;

  DataType type() const override;
  size_t serialize(void *buf) const override;
  size_t hash() const override;
  size_t size() const override;
  size_t byte_size() const override;
  const void* get_data() const override;

 private:
  const std::shared_ptr<char> data_;
  const size_t start_;
  const size_t end_;
};

class Output {
 public:
  Output(const std::shared_ptr<OutputData> y_hat,
         const std::vector<VersionedModelId> models_used);

  ~Output() = default;

  explicit Output() = default;
  Output(const Output &) = default;
  Output &operator=(const Output &) = default;

  Output(Output &&) = default;
  Output &operator=(Output &&) = default;

  bool operator==(const Output &rhs) const;
  bool operator!=(const Output &rhs) const;

  std::shared_ptr<OutputData> y_hat_;
  std::vector<VersionedModelId> models_used_;
};

namespace rpc {

class PredictionRequest {
 public:
  explicit PredictionRequest(DataType input_type);
  explicit PredictionRequest(std::vector<std::shared_ptr<Input>> inputs,
                             DataType input_type);

  // Disallow copy
  PredictionRequest(PredictionRequest &other) = delete;
  PredictionRequest &operator=(PredictionRequest &other) = delete;

  // move constructors
  PredictionRequest(PredictionRequest &&other) = default;
  PredictionRequest &operator=(PredictionRequest &&other) = default;

  void add_input(std::shared_ptr<Input> input);
  std::vector<ByteBuffer> serialize();

 private:
  void validate_input_type(std::shared_ptr<Input> &input) const;

  std::vector<std::shared_ptr<Input>> inputs_;
  DataType input_type_;
  size_t input_data_size_ = 0;
};

class PredictionResponse {
 public:
  PredictionResponse(const std::vector<std::shared_ptr<OutputData>> outputs);

  // Disallow copy
  PredictionResponse(PredictionResponse &other) = delete;
  PredictionResponse &operator=(PredictionResponse &other) = delete;

  // move constructors
  PredictionResponse(PredictionResponse &&other) = default;
  PredictionResponse &operator=(PredictionResponse &&other) = default;

  static PredictionResponse deserialize_prediction_response(
      DataType data_type, std::shared_ptr<void> &data);

  const std::vector<std::shared_ptr<OutputData>> outputs_;
};

}  // namespace rpc

class Response {
 public:
  ~Response() = default;

  Response(Query query, QueryId query_id, const long duration_micros,
           Output output, const bool is_default,
           const boost::optional<std::string> default_explanation);

  // default copy constructors
  Response(const Response &) = default;
  Response &operator=(const Response &) = default;

  // default move constructors
  Response(Response &&) = default;
  Response &operator=(Response &&) = default;

  Query query_;
  QueryId query_id_;
  long duration_micros_;
  Output output_;
  bool output_is_default_;
  boost::optional<std::string> default_explanation_;
};

}  // namespace clipper
namespace std {
template <>
struct hash<clipper::VersionedModelId> {
  typedef std::size_t result_type;
  std::size_t operator()(const clipper::VersionedModelId &vm) const {
    std::size_t seed = 0;
    boost::hash_combine(seed, vm.get_name());
    boost::hash_combine(seed, vm.get_id());
    return seed;
  }
};
}
#endif  // CLIPPER_LIB_DATATYPES_H
