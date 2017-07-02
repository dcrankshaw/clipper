#include <time.h>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/thread.hpp>
#include <cxxopts.hpp>

#include <clipper/constants.hpp>
#include <clipper/dag_executor.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>

using namespace clipper;

int main() {
  std::vector<VersionedModelId> nodes{
      VersionedModelId("source", "1"), VersionedModelId("m1", "1"),
      // VersionedModelId("m2", "1"), VersionedModelId("m3", "1"),
      VersionedModelId("sink", "1")};

  std::unordered_map<VersionedModelId, VersionedModelId> edges{
      {nodes[0], nodes[1]}, {nodes[1], nodes[2]},
      // {nodes[2], nodes[3]},
      // {nodes[3], nodes[4]}
  };

  DAGExecutor dag{nodes, edges};
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
