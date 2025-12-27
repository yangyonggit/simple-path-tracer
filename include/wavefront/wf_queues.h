#pragma once
#include <vector>
#include <cstdint>

namespace wf {

// Minimal CPU queue. CUDA version will replace this with device buffers + counters.
struct IndexQueue {
  std::vector<uint32_t> items;

  void clear() { items.clear(); }
  bool empty() const { return items.empty(); }
  size_t size() const { return items.size(); }

  void push(uint32_t v) { items.push_back(v); }
};

} // namespace wf
