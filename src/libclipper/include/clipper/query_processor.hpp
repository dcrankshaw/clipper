#ifndef CLIPPER_LIB_QUERY_PROCESSOR_H
#define CLIPPER_LIB_QUERY_PROCESSOR_H

#include <atomic>
#include <string>
#include <tuple>
#include <utility>

#include "datatypes.hpp"
#include "metrics.hpp"
#include "persistent_state.hpp"
#include "rpc_service.hpp"
#include "selection_policies.hpp"
#include "task_executor.hpp"
#include "timers.hpp"

namespace clipper {

const std::string LOGGING_TAG_QUERY_PROCESSOR = "QUERYPROCESSOR";

class QueryProcessor {
 public:
  ~QueryProcessor() = default;

  QueryProcessor();

  // Disallow copies
  QueryProcessor(const QueryProcessor& other) = delete;
  QueryProcessor& operator=(const QueryProcessor& other) = delete;

  // Default move constructor and assignment.
  QueryProcessor(QueryProcessor&& other) = default;
  QueryProcessor& operator=(QueryProcessor&& other) = default;

  void predict(Query query, std::function<void(Response)>&& on_response_callback);

  std::shared_ptr<StateDB> get_state_table() const;

 private:
  std::atomic<long> query_counter_{0};
  std::shared_ptr<StateDB> state_db_;
  std::shared_ptr<SelectionState> selection_state_;
  TaskExecutor task_executor_;
  TimerSystem<HighPrecisionClock> timer_system_{HighPrecisionClock()};
  // This is a heteregenous container of different instances of selection
  // policy. The key is the name of the specific selection policy, the value is
  // an instance of that policy. All SelectionPolicy implementations (derived
  // classes) should be stateless so there should be no issues with re-using the
  // same instance for different applications or users.
  std::unordered_map<std::string, std::shared_ptr<SelectionPolicy>>
      selection_policies_;
  std::shared_ptr<metrics::Meter> request_rate_;
};

}  // namespace clipper

#endif
