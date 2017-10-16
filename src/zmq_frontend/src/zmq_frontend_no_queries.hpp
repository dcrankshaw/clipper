#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <algorithm>

#include <folly/futures/Future.h>
#include <wangle/concurrent/IOThreadPoolExecutor.h>

#include "frontend_rpc_service.hpp"

#include <clipper/config.hpp>
#include <clipper/constants.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/exceptions.hpp>
#include <clipper/json_util.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/query_processor.hpp>
#include <clipper/redis.hpp>

using clipper::Response;
using clipper::FeedbackAck;
using clipper::VersionedModelId;
using clipper::DataType;
using clipper::Input;
using clipper::Output;
using clipper::OutputData;
using clipper::Query;
using clipper::Feedback;
using clipper::FeedbackQuery;
using clipper::json::json_parse_error;
using clipper::json::json_semantic_error;
using clipper::redis::labels_to_str;

namespace zmq_frontend {

const std::string LOGGING_TAG_RPC_FRONTEND = "RPC FRONTEND";
const std::string GET_METRICS = "^/metrics$";

const char* PREDICTION_RESPONSE_KEY_QUERY_ID = "query_id";
const char* PREDICTION_RESPONSE_KEY_OUTPUT = "output";
const char* PREDICTION_RESPONSE_KEY_USED_DEFAULT = "default";
const char* PREDICTION_RESPONSE_KEY_DEFAULT_EXPLANATION = "default_explanation";
const char* PREDICTION_ERROR_RESPONSE_KEY_ERROR = "error";
const char* PREDICTION_ERROR_RESPONSE_KEY_CAUSE = "cause";

const std::string PREDICTION_ERROR_NAME_REQUEST = "Request error";
const std::string PREDICTION_ERROR_NAME_JSON = "Json error";
const std::string PREDICTION_ERROR_NAME_QUERY_PROCESSING =
    "Query processing error";

/* Generate a user-facing error message containing the exception
 * content and the expected JSON schema. */
std::string json_error_msg(const std::string& exception_msg,
                           const std::string& expected_schema) {
  std::stringstream ss;
  ss << "Error parsing JSON: " << exception_msg << ". "
     << "Expected JSON schema: " << expected_schema;
  return ss.str();
}

class AppMetrics {
 public:
  explicit AppMetrics(std::string app_name)
      : app_name_(app_name),
        latency_(
            clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
                "app:" + app_name + ":prediction_latency", "microseconds",
                4096)),
        throughput_(
            clipper::metrics::MetricsRegistry::get_metrics().create_meter(
                "app:" + app_name + ":prediction_throughput")),
        num_predictions_(
            clipper::metrics::MetricsRegistry::get_metrics().create_counter(
                "app:" + app_name + ":num_predictions")),
        default_pred_ratio_(
            clipper::metrics::MetricsRegistry::get_metrics()
                .create_ratio_counter("app:" + app_name +
                                      ":default_prediction_ratio")) {}
  ~AppMetrics() = default;

  AppMetrics(const AppMetrics&) = default;

  AppMetrics& operator=(const AppMetrics&) = default;

  AppMetrics(AppMetrics&&) = default;
  AppMetrics& operator=(AppMetrics&&) = default;

  std::string app_name_;
  std::shared_ptr<clipper::metrics::Histogram> latency_;
  std::shared_ptr<clipper::metrics::Meter> throughput_;
  std::shared_ptr<clipper::metrics::Counter> num_predictions_;
  std::shared_ptr<clipper::metrics::RatioCounter> default_pred_ratio_;
};

template <typename T, class N>
std::vector<std::shared_ptr<Input>> get_primitive_inputs(
    int message_size, int input_len, DataType type, std::vector<T> data_vector,
    std::vector<std::shared_ptr<N>> input_vector) {
  input_vector.clear();
  std::vector<std::shared_ptr<Input>> generic_input_vector;
  for (int k = 0; k < message_size; ++k) {
    for (int j = 0; j < input_len; ++j) {
      if (type == DataType::Bytes) {
        uint8_t *bytes = reinterpret_cast<uint8_t *>(&j);
        for (int i = 0; i < (int)(sizeof(int) / sizeof(uint8_t)); i++) {
          data_vector.push_back(*(bytes + i));
        }
      } else {
        data_vector.push_back(static_cast<T>(j));
      }
    }
    std::shared_ptr<T> input_data(
        static_cast<T *>(malloc(sizeof(T) * data_vector.size())), free);
    memcpy(input_data.get(), data_vector.data(),
           data_vector.size() * sizeof(T));

    std::shared_ptr<N> input =
        std::make_shared<N>(input_data, data_vector.size());
    generic_input_vector.push_back(std::dynamic_pointer_cast<Input>(input));
    data_vector.clear();
  }
  return generic_input_vector;
}

std::vector<std::shared_ptr<Input>> generate_floats_inputs(int message_size) {
  std::vector<float> type_vec;
  std::vector<std::shared_ptr<FloatVector>> input_vec;
  std::vector<std::shared_ptr<Input>> inputs = get_primitive_inputs(
      message_size, 299*299*3, DataType::Floats, type_vec, input_vec);
  return inputs;
}

class ServerImplNoQueries {
 public:
  ServerImplNoQueries(const std::string ip, int send_port, int recv_port)
      : rpc_service_(std::make_shared<FrontendRPCService>()),
        futures_executor_(std::make_shared<wangle::IOThreadPoolExecutor>(6)),
        query_processor_(futures_executor_) {
    // Init Clipper stuff

    // Start the frontend rpc service
    rpc_service_->start(ip, send_port, recv_port);

    // std::string server_address = address + std::to_string(portno);
    clipper::Config& conf = clipper::get_config();
    while (!redis_connection_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      clipper::log_error(LOGGING_TAG_RPC_FRONTEND,
                         "Query frontend failed to connect to Redis",
                         "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    while (!redis_subscriber_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      clipper::log_error(LOGGING_TAG_RPC_FRONTEND,
                         "Query frontend subscriber failed to connect to Redis",
                         "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // server_.add_endpoint(GET_METRICS, "GET",
    //                      [](std::shared_ptr<HttpServer::Response> response,
    //                         std::shared_ptr<HttpServer::Request>
    //                         #<{(|request|)}>#) {
    //                        clipper::metrics::MetricsRegistry& registry =
    //                            clipper::metrics::MetricsRegistry::get_metrics();
    //                        std::string metrics_report =
    //                            registry.report_metrics();
    //                        clipper::log_info(LOGGING_TAG_RPC_FRONTEND,
    //                                          "METRICS", metrics_report);
    //                        respond_http(metrics_report, "200 OK", response);
    //                      });

    clipper::redis::subscribe_to_application_changes(
        redis_subscriber_,
        [this](const std::string& key, const std::string& event_type) {
          clipper::log_info_formatted(
              LOGGING_TAG_RPC_FRONTEND,
              "APPLICATION EVENT DETECTED. Key: {}, event_type: {}", key,
              event_type);
          if (event_type == "hset") {
            std::string name = key;
            clipper::log_info_formatted(LOGGING_TAG_RPC_FRONTEND,
                                        "New application detected: {}", key);
            auto app_info =
                clipper::redis::get_application_by_key(redis_connection_, key);
            DataType input_type =
                clipper::parse_input_type(app_info["input_type"]);
            std::string policy = app_info["policy"];
            std::string default_output = app_info["default_output"];
            int latency_slo_micros = std::stoi(app_info["latency_slo_micros"]);
            add_application(name, input_type, policy, default_output,
                            latency_slo_micros);
          }
        });

    clipper::redis::subscribe_to_model_link_changes(
        redis_subscriber_,
        [this](const std::string& key, const std::string& event_type) {
          std::string app_name = key;
          clipper::log_info_formatted(
              LOGGING_TAG_RPC_FRONTEND,
              "APP LINKS EVENT DETECTED. App name: {}, event_type: {}",
              app_name, event_type);
          if (event_type == "sadd") {
            clipper::log_info_formatted(LOGGING_TAG_RPC_FRONTEND,
                                        "New model link detected for app: {}",
                                        app_name);
            auto linked_model_names =
                clipper::redis::get_linked_models(redis_connection_, app_name);
            set_linked_models_for_app(app_name, linked_model_names);
          }
        });

    clipper::redis::subscribe_to_model_version_changes(
        redis_subscriber_,
        [this](const std::string& key, const std::string& event_type) {
          clipper::log_info_formatted(
              LOGGING_TAG_RPC_FRONTEND,
              "MODEL VERSION CHANGE DETECTED. Key: {}, event_type: {}", key,
              event_type);
          if (event_type == "set") {
            std::string model_name = key;
            boost::optional<std::string> new_version =
                clipper::redis::get_current_model_version(redis_connection_,
                                                          key);
            if (new_version) {
              std::unique_lock<std::mutex> l(current_model_versions_mutex_);
              current_model_versions_[key] = *new_version;
            } else {
              clipper::log_error_formatted(
                  LOGGING_TAG_RPC_FRONTEND,
                  "Model version change for model {} was invalid.", key);
            }
          }
        });

    // Read from Redis configuration tables and update models/applications.
    // (1) Iterate through applications and set up predict/update endpoints.
    std::vector<std::string> app_names =
        clipper::redis::get_all_application_names(redis_connection_);
    for (std::string app_name : app_names) {
      auto app_info =
          clipper::redis::get_application_by_key(redis_connection_, app_name);

      auto linked_model_names =
          clipper::redis::get_linked_models(redis_connection_, app_name);
      set_linked_models_for_app(app_name, linked_model_names);

      DataType input_type = clipper::parse_input_type(app_info["input_type"]);
      std::string policy = app_info["policy"];
      std::string default_output = app_info["default_output"];
      int latency_slo_micros = std::stoi(app_info["latency_slo_micros"]);

      add_application(app_name, input_type, policy, default_output,
                      latency_slo_micros);
    }
    if (app_names.size() > 0) {
      clipper::log_info_formatted(
          LOGGING_TAG_RPC_FRONTEND,
          "Found {} existing applications registered in Clipper: {}.",
          app_names.size(), labels_to_str(app_names));
    }
    // (2) Update current_model_versions_ with (model, version) pairs.
    std::vector<std::string> model_names =
        clipper::redis::get_all_model_names(redis_connection_);
    // Record human-readable model names for logging
    std::vector<std::string> model_names_with_version;
    for (std::string model_name : model_names) {
      auto model_version = clipper::redis::get_current_model_version(
          redis_connection_, model_name);
      if (model_version) {
        std::unique_lock<std::mutex> l(current_model_versions_mutex_);
        current_model_versions_[model_name] = *model_version;
        model_names_with_version.push_back(model_name + "@" + *model_version);
      } else {
        clipper::log_error_formatted(
            LOGGING_TAG_RPC_FRONTEND,
            "Found model {} with missing current version.", model_name);
        throw std::runtime_error("Invalid model version");
      }
    }
    if (model_names.size() > 0) {
      clipper::log_info_formatted(
          LOGGING_TAG_RPC_FRONTEND, "Found {} models deployed to Clipper: {}.",
          model_names.size(), labels_to_str(model_names_with_version));
    }
  }

  ~ServerImplNoQueries() {
    redis_connection_.disconnect();
    redis_subscriber_.disconnect();
    rpc_service_->stop();
  }

  void set_linked_models_for_app(std::string name,
                                 std::vector<std::string> models) {
    std::unique_lock<std::mutex> l(linked_models_for_apps_mutex_);
    linked_models_for_apps_[name] = models;
  }

  std::vector<std::string> get_linked_models_for_app(std::string name) {
    std::unique_lock<std::mutex> l(linked_models_for_apps_mutex_);
    return linked_models_for_apps_[name];
  }

  void add_application(std::string name, DataType input_type,
                       std::string policy, std::string default_output,
                       long latency_slo_micros) {
    // TODO: QueryProcessor should handle this. We need to decide how the
    // default output fits into the generic selection policy API. Do all
    // selection policies have a default output?

    // Initialize selection state for this application
    if (policy == clipper::DefaultOutputSelectionPolicy::get_name()) {
      clipper::DefaultOutputSelectionPolicy p;
      std::shared_ptr<char> default_output_content(
          static_cast<char*>(malloc(sizeof(default_output))), free);
      memcpy(default_output_content.get(), default_output.data(),
             default_output.size());
      clipper::Output parsed_default_output(
          std::make_shared<clipper::StringOutput>(default_output_content, 0,
                                                  default_output.size()),
          {});
      auto init_state = p.init_state(parsed_default_output);
      clipper::StateKey state_key{name, clipper::DEFAULT_USER_ID, 0};
      query_processor_.get_state_table()->put(state_key,
                                              p.serialize(init_state));
    }

    AppMetrics app_metrics(name);

    predict_func_ = [
      this, name, application_input_type = input_type, policy,
      latency_slo_micros, app_metrics
    ](std::shared_ptr<Input> input) {
      try {
        std::vector<std::string> models = get_linked_models_for_app(name);
        std::vector<VersionedModelId> versioned_models;
        {
          std::unique_lock<std::mutex> l(current_model_versions_mutex_);
          for (auto m : models) {
            auto version = current_model_versions_.find(m);
            if (version != current_model_versions_.end()) {
              versioned_models.emplace_back(m, version->second);
            }
          }
        }

        long uid = 0;
        // int request_id = std::get<1>(request);
        // int client_id = std::get<2>(request);
        auto prediction = query_processor_.predict(
            Query{name, uid, input, latency_slo_micros, policy,
                  versioned_models});

        prediction.via(futures_executor_.get())
            .then([this, app_metrics](Response r) {
              // Update metrics
              if (r.output_is_default_) {
                app_metrics.default_pred_ratio_->increment(1, 1);
              } else {
                app_metrics.default_pred_ratio_->increment(0, 1);
                if (r.output_.y_hat_->type() == clipper::DataType::Strings) {
                  std::string debugstr = std::string(
                      static_cast<const char *>(r.output_.y_hat_->get_data()),
                      r.output_.y_hat_->byte_size());
                  clipper::log_info_formatted(clipper::LOGGING_TAG_CLIPPER,
                      "Responding with: {}", debugstr);
                }
              }
              app_metrics.latency_->insert(r.duration_micros_);
              app_metrics.num_predictions_->increment(1);

              // rpc_service_->send_response(
              //     std::make_tuple(std::move(r.output_), request_id, client_id));
            })
            .onError([](const std::exception& e) {
              clipper::log_error_formatted(clipper::LOGGING_TAG_CLIPPER,
                                           "Unexpected error: {}", e.what());
              // TODO(czumar): Do something here!
              return;
            });
      } catch (const std::invalid_argument& e) {
        // This invalid argument exception is most likely the propagation of an
        // exception thrown
        // when Rapidjson attempts to parse an invalid json schema
        std::string error_response = get_prediction_error_response_content(
            PREDICTION_ERROR_NAME_JSON, e.what());
        // TODO(czumar): Do something here!
      } catch (const clipper::PredictError& e) {
        std::string error_response = get_prediction_error_response_content(
            PREDICTION_ERROR_NAME_QUERY_PROCESSING, e.what());
        // TODO(czumar): Do something here!
      }
    };

    predict_func_ready_ = true;
    // rpc_service_->add_application(name, predict_fn);
  }

  void run_benchmark() {
    std::vector<std::shared_ptr<Input>> inputs = generate_floats_inputs(500);
    while (!predict_func_ready_) {
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    while (true) {
      for (int i = 0; i < inputs.size(); ++i) {
        if (i % 10 == 0) {
          predict_func_(inputs[i]);
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
    }
  }

  /**
   * Obtains user-readable http response content for a query
   * that could not be completed due to an error
   */
  static const std::string get_prediction_error_response_content(
      const std::string error_name, const std::string error_msg) {
    std::stringstream ss;
    ss << error_name << ": " << error_msg;
    return ss.str();
  }

  /**
   * Returns a copy of the map containing current model names and versions.
   */
  std::unordered_map<std::string, std::string> get_current_model_versions() {
    return current_model_versions_;
  }

  std::string get_metrics() const {
    clipper::metrics::MetricsRegistry& registry =
        clipper::metrics::MetricsRegistry::get_metrics();
    std::string metrics_report = registry.report_metrics();
    // clipper::log_info(LOGGING_TAG_RPC_FRONTEND, "METRICS", metrics_report);
    return metrics_report;
  }

  std::function<void(std::shared_ptr<Input>)> predict_func_;

 private:
  std::shared_ptr<FrontendRPCService> rpc_service_;
  redox::Redox redis_connection_;
  redox::Subscriber redis_subscriber_;
  std::mutex current_model_versions_mutex_;
  std::unordered_map<std::string, std::string> current_model_versions_;

  std::mutex linked_models_for_apps_mutex_;
  std::unordered_map<std::string, std::vector<std::string>>
      linked_models_for_apps_;

  std::shared_ptr<wangle::IOThreadPoolExecutor> futures_executor_;
  clipper::QueryProcessor query_processor_;
  std::atomic<bool> predict_func_ready_{false};
  
};

}  // namespace zmq_frontend
