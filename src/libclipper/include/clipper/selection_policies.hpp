#ifndef CLIPPER_LIB_SELECTION_POLICIES_H
#define CLIPPER_LIB_SELECTION_POLICIES_H

#include <map>
#include <memory>
#include <unordered_map>

#include "datatypes.hpp"
#include "task_executor.hpp"

namespace clipper {

const std::string LOGGING_TAG_SELECTION_POLICY = "SELECTIONPOLICY";

class SelectionState {
 public:
  SelectionState() = default;
  SelectionState(const SelectionState&) = default;
  SelectionState& operator=(const SelectionState&) = default;
  SelectionState(SelectionState&&) = default;
  SelectionState& operator=(SelectionState&&) = default;
  /**
   * Constructor to create a SelectionState object from serialized
   * representation.
   */
  virtual ~SelectionState();
  virtual std::string get_debug_string() const = 0;
};

class SelectionPolicy {
  // Note that we need to use pointers to the SelectionState because
  // C++ doesn't let you have an object with an abstract type on the stack
  // because it doesn't know how big it is. It must be on heap.
  // We use shared_ptr instead of unique_ptr so that we can do pointer
  // casts down the inheritance hierarchy.

 public:
  SelectionPolicy() = default;
  SelectionPolicy(const SelectionPolicy&) = default;
  SelectionPolicy& operator=(const SelectionPolicy&) = default;
  SelectionPolicy(SelectionPolicy&&) = default;
  SelectionPolicy& operator=(SelectionPolicy&&) = default;
  virtual ~SelectionPolicy();
  // virtual SelectionState initialize(
  //     const std::vector<VersionedModelId>& candidate_models) const = 0;
  // virtual std::shared_ptr<SelectionState> update_candidate_models(
  //     std::shared_ptr<SelectionState> state,
  //     const std::vector<VersionedModelId>& candidate_models) const = 0;

  // Used to identify a unique selection policy instance. For example,
  // if using a bandit-algorithm that does not tolerate variable-armed
  // bandits, one could hash the candidate models to identify
  // which policy instance corresponds to this exact set of arms.
  // Similarly, it provides flexibility in how to deal with different
  // versions of the same arm (different versions of same model).
  virtual long hash_models(
      const std::vector<VersionedModelId>& /*models*/) const {
    return 0;
  }

  // Query Pre-processing: select models and generate tasks
  virtual std::vector<PredictTask> select_predict_tasks(
      std::shared_ptr<SelectionState> state, Query query,
      long query_id) const = 0;

  virtual Output combine_predictions(
      const std::shared_ptr<SelectionState>& state, Query query,
      std::vector<Output> predictions) const = 0;

  /// When feedback is received, the selection policy can choose
  /// to schedule both feedback and prediction tasks. Prediction tasks
  /// can be used to get y_hat for e.g. updating a bandit algorithm,
  /// while feedback tasks can be used to optionally propogate feedback
  /// into the model containers.
  virtual std::pair<std::vector<PredictTask>, std::vector<FeedbackTask>>
  select_feedback_tasks(const std::shared_ptr<SelectionState>& state,
                        FeedbackQuery query, long query_id) const = 0;

  /// This method will be called if at least one PredictTask
  /// was scheduled for this piece of feedback. This method
  /// is guaranteed to be called sometime after all the predict
  /// tasks scheduled by `select_feedback_tasks` complete.
  virtual std::shared_ptr<SelectionState> process_feedback(
      std::shared_ptr<SelectionState> state, Feedback feedback,
      std::vector<Output> predictions) const = 0;

  virtual std::shared_ptr<SelectionState> deserialize(
      std::string serialized_state) const = 0;
  virtual std::string serialize(
      std::shared_ptr<SelectionState> state) const = 0;
};

////////////////////////////////////////////////////////////////////

class DefaultOutputSelectionState : public SelectionState {
 public:
  DefaultOutputSelectionState() = default;
  DefaultOutputSelectionState(const DefaultOutputSelectionState&) = default;
  DefaultOutputSelectionState& operator=(const DefaultOutputSelectionState&) =
      default;
  DefaultOutputSelectionState(DefaultOutputSelectionState&&) = default;
  DefaultOutputSelectionState& operator=(DefaultOutputSelectionState&&) =
      default;
  explicit DefaultOutputSelectionState(Output default_output);
  explicit DefaultOutputSelectionState(std::string serialized_state);
  ~DefaultOutputSelectionState() = default;
  std::string serialize() const;
  std::string get_debug_string() const override;
  Output default_output_;

 private:
  static Output deserialize(std::string serialized_state);
};

class DefaultOutputSelectionPolicy : public SelectionPolicy {
 public:
  DefaultOutputSelectionPolicy() = default;
  DefaultOutputSelectionPolicy(const DefaultOutputSelectionPolicy&) = default;
  DefaultOutputSelectionPolicy& operator=(const DefaultOutputSelectionPolicy&) =
      default;
  DefaultOutputSelectionPolicy(DefaultOutputSelectionPolicy&&) = default;
  DefaultOutputSelectionPolicy& operator=(DefaultOutputSelectionPolicy&&) =
      default;
  ~DefaultOutputSelectionPolicy() = default;
  std::shared_ptr<SelectionState> init_state(Output default_output) const;

  std::vector<PredictTask> select_predict_tasks(
      std::shared_ptr<SelectionState> state, Query query,
      long query_id) const override;

  Output combine_predictions(const std::shared_ptr<SelectionState>& state,
                             Query query,
                             std::vector<Output> predictions) const override;

  std::pair<std::vector<PredictTask>, std::vector<FeedbackTask>>
  select_feedback_tasks(const std::shared_ptr<SelectionState>& state,
                        FeedbackQuery query, long query_id) const override;

  std::shared_ptr<SelectionState> process_feedback(
      std::shared_ptr<SelectionState> state, Feedback feedback,
      std::vector<Output> predictions) const override;

  std::shared_ptr<SelectionState> deserialize(
      std::string serialized_state) const override;
  std::string serialize(std::shared_ptr<SelectionState> state) const override;
};

}  // namespace clipper

#endif  // CLIPPER_LIB_SELECTION_POLICIES_H
