#include <cassert>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstring>
#include <tuple>

#include <folly/futures/Future.h>

#include <clipper/config.hpp>
#include <clipper/constants.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/exceptions.hpp>
#include <clipper/json_util.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/query_processor.hpp>
#include <clipper/redis.hpp>

#include <grpc++/grpc++.h>
#include <grpc++/server.h>
#include <grpc/grpc.h>

#include "clipper_frontend.grpc.pb.h"

// #include <server_http.hpp>

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
// using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;
using namespace clipper::grpc;

namespace rpc_frontend {

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

class ServerRpcContext {
 public:
  ServerRpcContext(
      std::function<void(grpc::ServerContext*, PredictRequest*,
                         grpc::ServerAsyncResponseWriter<PredictResponse>*,
                         void*)>
          request_method,
      std::function<void(std::string, ServerRpcContext*)> invoke_method)
      : status_(grpc::Status::OK),
        srv_ctx_(new grpc::ServerContext),
        next_state_(&ServerRpcContext::invoker),
        request_method_(request_method),
        invoke_method_(invoke_method),
        response_writer_(srv_ctx_.get()) {
    request_method_(srv_ctx_.get(), &req_, &response_writer_,
                    ServerRpcContext::tag(this));
  }
  ~ServerRpcContext() {}

  bool RunNextState(bool ok) { return (this->*next_state_)(ok); }

  void Reset() {
    srv_ctx_.reset(new grpc::ServerContext);
    req_ = PredictRequest();
    response_writer_ =
        grpc::ServerAsyncResponseWriter<PredictResponse>(srv_ctx_.get());

    status_ = grpc::Status::OK;
    // Then request the method
    next_state_ = &ServerRpcContext::invoker;
    request_method_(srv_ctx_.get(), &req_, &response_writer_,
                    ServerRpcContext::tag(this));
  }

  static void* tag(ServerRpcContext* func) {
    return reinterpret_cast<void*>(func);
  }
  static ServerRpcContext* detag(void* tag) {
    return reinterpret_cast<ServerRpcContext*>(tag);
  }

  void send_response() {
    // Have the response writer work and invoke on_finish when done
    next_state_ = &ServerRpcContext::finisher;
    response_writer_.Finish(response_, status_, ServerRpcContext::tag(this));
  }

  PredictRequest req_;
  PredictResponse response_;
  grpc::Status status_;

 private:
  bool finisher(bool) { return false; }

  bool invoker(bool ok) {
    if (!ok) {
      return false;
    }
    // Call the RPC processing function
    invoke_method_(req_.application(), this);
    return true;
  }

  std::unique_ptr<grpc::ServerContext> srv_ctx_;
  bool (ServerRpcContext::*next_state_)(bool);
  std::function<void(grpc::ServerContext*, PredictRequest*,
                     grpc::ServerAsyncResponseWriter<PredictResponse>*, void*)>
      request_method_;
  std::function<void(std::string, ServerRpcContext*)> invoke_method_;
  grpc::ServerAsyncResponseWriter<PredictResponse> response_writer_;
};

class RequestHandler {
 public:
  RequestHandler() : query_processor_(), active_(true) {
    // Init Clipper stuff

    request_throughput_ = clipper::metrics::MetricsRegistry::get_metrics().create_meter("grpc_request_throughput");

    metrics_thread_ = std::thread([this]() {
      while(active_) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::string metrics_report = clipper::metrics::MetricsRegistry::get_metrics().report_metrics();
        clipper::log_debug("METRICS", metrics_report);
      }
    });

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

  ~RequestHandler() {
    redis_connection_.disconnect();
    redis_subscriber_.disconnect();
    active_ = false;
    metrics_thread_.join();
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

    auto predict_fn = [this, name, application_input_type = input_type, policy, latency_slo_micros,
                       app_metrics](ServerRpcContext* rpc_context) {
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

        // std::string app_name = request.application();

        DataType request_input_type = static_cast<DataType>(rpc_context->req_.data_type());
        if(request_input_type != application_input_type) {
          clipper::log_error_formatted(
              LOGGING_TAG_RPC_FRONTEND,
              "Received prediction request with inputs of type: {} for application expecting inputs of type: {}",
              clipper::get_readable_input_type(request_input_type),
              clipper::get_readable_input_type(application_input_type));
          // TODO(czumar): Return bad request response
          return;
        }

        std::shared_ptr<clipper::Input> input;
        switch(request_input_type) {
          case DataType::Bytes:
            input = std::make_shared<clipper::ByteVector>(
                reinterpret_cast<const uint8_t*>(rpc_context->req_.byte_data().data().data()),
                rpc_context->req_.byte_data().data().size());
            break;
          case DataType::Ints:
            input = std::make_shared<clipper::IntVector>(
                rpc_context->req_.int_data().data().data(), rpc_context->req_.int_data().data().size());
            break;
          case DataType::Floats:
            input = std::make_shared<clipper::FloatVector>(
                rpc_context->req_.float_data().data().data(), rpc_context->req_.float_data().data().size());
            break;
          case DataType::Doubles:
            input = std::make_shared<clipper::DoubleVector>(
                rpc_context->req_.double_data().data().data(), rpc_context->req_.double_data().data().size());
            break;
          case DataType::Strings:
            input = std::make_shared<clipper::SerializableString>(
                rpc_context->req_.string_data().data().data(), rpc_context->req_.string_data().data().size());
            break;
          case DataType::Invalid:
          default: {
            std::stringstream ss;
            ss << "Attempted to create an input with invalid type: " << get_readable_input_type(request_input_type);
            throw std::runtime_error(ss.str());
          } break;
        }

        long uid = 0;
        folly::Future<clipper::Response> prediction =
            query_processor_.predict(Query{name, uid, input, latency_slo_micros,
                                           policy, versioned_models});

        request_throughput->mark(1);

        prediction
          .then([app_metrics, rpc_context](Response r) {
          // Update metrics
          if (r.output_is_default_) {
            app_metrics.default_pred_ratio_->increment(1, 1);
          } else {
            app_metrics.default_pred_ratio_->increment(0, 1);
          }
          app_metrics.latency_->insert(r.duration_micros_);
          app_metrics.num_predictions_->increment(1);
          app_metrics.throughput_->mark(1);

          PredictResponse &response = rpc_context->response_;
          response.set_has_error(false);
          std::shared_ptr<OutputData> output_data = r.output_.y_hat_;
          response.set_data_type(static_cast<int>(output_data->type()));

          switch(output_data->type()) {
            case DataType::Bytes:
              response.mutable_byte_data()->set_data(output_data->get_data(), output_data->size());
              break;
            case DataType::Ints: {
              response.mutable_int_data()->mutable_data()->Resize(output_data->byte_size(), 0);
              output_data->serialize(response.mutable_int_data()->mutable_data()->mutable_data());
            } break;
            case DataType::Floats: {
              response.mutable_float_data()->mutable_data()->Resize(output_data->size(), 0);
              output_data->serialize(response.mutable_float_data()->mutable_data()->mutable_data());
            } break;
            case DataType::Strings: {
              response.mutable_string_data()->set_data(
                  static_cast<const char*>(output_data->get_data()), output_data->size());
            } break;
            case DataType::Invalid:
            default: {
              std::stringstream ss;
              ss << "Received a prediction response with an invalid output type: "
                 << get_readable_input_type(output_data->type());
              throw std::runtime_error(ss.str());
            } break;
          }

          response.set_default_explanation(r.default_explanation_.get_value_or(""));
          response.set_is_default(r.output_is_default_);

          rpc_context->send_response();
          })
        .onError([rpc_context](const std::exception& e) {
            clipper::log_error_formatted(clipper::LOGGING_TAG_CLIPPER,
                "Unexpected error: {}", e.what());
            // TODO: Use grpc status
            rpc_context->response_.set_has_error(true);
            rpc_context->response_.set_error("An unexpected error occurred!");
            rpc_context->send_response();
            return;
        });
      } catch (const std::invalid_argument& e) {
        // This invalid argument exception is most likely the propagation of an
        // exception thrown
        // when Rapidjson attempts to parse an invalid json schema
        std::string error_response = get_prediction_error_response_content(
            PREDICTION_ERROR_NAME_JSON, e.what());
        rpc_context->response_.set_has_error(true);
        rpc_context->response_.set_error(error_response);
        rpc_context->send_response();
      } catch (const clipper::PredictError& e) {
        std::string error_response = get_prediction_error_response_content(
            PREDICTION_ERROR_NAME_QUERY_PROCESSING, e.what());
        rpc_context->response_.set_has_error(true);
        rpc_context->response_.set_error(error_response);
        rpc_context->send_response();
      }
    };

    std::unique_lock<std::mutex> l(app_predict_functions_mutex_);
    app_predict_functions_.emplace(name, predict_fn);
  }

  void predict(std::string app_name, ServerRpcContext* rpc_context) {
    std::unique_lock<std::mutex> l(app_predict_functions_mutex_);
    auto search = app_predict_functions_.find(app_name);
    if (search != app_predict_functions_.end()) {
      l.unlock();
      search->second(rpc_context);
    } else {
      l.unlock();
      std::string error_response = get_prediction_error_response_content(
          PREDICTION_ERROR_NAME_REQUEST, "No registered application with name: " + app_name);
      rpc_context->response_.set_has_error(true);
      rpc_context->response_.set_error(error_response);
      rpc_context->send_response();
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

 private:
  // HttpServer http_server_;
  clipper::QueryProcessor query_processor_;
  redox::Redox redis_connection_;
  redox::Subscriber redis_subscriber_;
  std::mutex current_model_versions_mutex_;
  std::unordered_map<std::string, std::string> current_model_versions_;

  std::mutex linked_models_for_apps_mutex_;
  std::unordered_map<std::string, std::vector<std::string>>
      linked_models_for_apps_;

  std::shared_ptr<clipper::metrics::Meter> request_throughput_;

  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  Predict::AsyncService service_;
  std::unique_ptr<grpc::Server> rpc_server_;
  std::mutex app_predict_functions_mutex_;
  std::unordered_map<std::string, std::function<void(ServerRpcContext*)>>
      app_predict_functions_;
  std::atomic_bool active_;
  std::thread metrics_thread_;
};

class ServerImpl {
 public:
  ServerImpl(std::string address, int portno, int num_threads)
      : handler_(new RequestHandler{}) {
    std::string server_address = address + ":" + std::to_string(portno);

    grpc::ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
    for (int i = 0; i < num_threads; ++i) {
      srv_cqs_.emplace_back(builder.AddCompletionQueue());
    }

    server_ = builder.BuildAndStart();

    auto process_func = [this](std::string app_name,
                               ServerRpcContext* context) {
      handler_->predict(app_name, context);
    };

    for (int i = 0; i < 1000; ++i) {
      for (int j = 0; j < num_threads; j++) {
        auto request_func = [j, this](
            grpc::ServerContext* ctx, PredictRequest* request,
            grpc::ServerAsyncResponseWriter<PredictResponse>* responder,
            void* tag) {
          service_.RequestPredict(ctx, request, responder,
                                        srv_cqs_[j].get(), srv_cqs_[j].get(),
                                        tag);
        };
        contexts_.emplace_back(
            new ServerRpcContext(request_func, process_func));
      }
    }

    for (int i = 0; i < num_threads; i++) {
      shutdown_state_.emplace_back(new PerThreadShutdownState());
      threads_.emplace_back(&ServerImpl::ThreadFunc, this, i);
    }
  }

  ~ServerImpl() {
    for (auto ss = shutdown_state_.begin(); ss != shutdown_state_.end(); ++ss) {
      std::lock_guard<std::mutex> lock((*ss)->mutex);
      (*ss)->shutdown = true;
    }
    std::thread shutdown_thread(&ServerImpl::ShutdownThreadFunc, this);
    for (auto cq = srv_cqs_.begin(); cq != srv_cqs_.end(); ++cq) {
      (*cq)->Shutdown();
    }
    for (auto thr = threads_.begin(); thr != threads_.end(); thr++) {
      thr->join();
    }
    for (auto cq = srv_cqs_.begin(); cq != srv_cqs_.end(); ++cq) {
      bool ok;
      void* got_tag;
      while ((*cq)->Next(&got_tag, &ok))
        ;
    }
    shutdown_thread.join();
  }

  std::string get_metrics() const {
    clipper::metrics::MetricsRegistry& registry =
        clipper::metrics::MetricsRegistry::get_metrics();
    std::string metrics_report = registry.report_metrics();
    // clipper::log_info(LOGGING_TAG_RPC_FRONTEND, "METRICS", metrics_report);
    return metrics_report;
  }

 private:
  void ShutdownThreadFunc() {
    // TODO (vpai): Remove this deadline and allow Shutdown to finish properly
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
    server_->Shutdown(deadline);
  }

  void ThreadFunc(int thread_idx) {
    // Wait until work is available or we are shutting down
    bool ok;
    void* got_tag;
    while (srv_cqs_[thread_idx]->Next(&got_tag, &ok)) {
      ServerRpcContext* ctx = ServerRpcContext::detag(got_tag);
      // The tag is a pointer to an RPC context to invoke
      // Proceed while holding a lock to make sure that
      // this thread isn't supposed to shut down
      std::lock_guard<std::mutex> l(shutdown_state_[thread_idx]->mutex);
      if (shutdown_state_[thread_idx]->shutdown) {
        return;
      }
      const bool still_going = ctx->RunNextState(ok);
      // if this RPC context is done, refresh it
      if (!still_going) {
        ctx->Reset();
      }
    }
    return;
  }

  std::vector<std::unique_ptr<grpc::ServerCompletionQueue>> srv_cqs_;
  Predict::AsyncService service_;
  std::unique_ptr<grpc::Server> server_;
  std::vector<std::thread> threads_;
  std::vector<std::unique_ptr<ServerRpcContext>> contexts_;
  std::unique_ptr<RequestHandler> handler_;

  struct PerThreadShutdownState {
    mutable std::mutex mutex;
    bool shutdown;
    PerThreadShutdownState() : shutdown(false) {}
  };

  std::vector<std::unique_ptr<PerThreadShutdownState>> shutdown_state_;
};

}  // namespace query_frontend
