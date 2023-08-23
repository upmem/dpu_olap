#pragma once

#include <stdint.h>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace timer {

class Timer {
 public:
  explicit Timer(uint32_t nb_ranks);
  void Start(uint32_t id=0);
  void Stop(uint32_t id=0);
  std::chrono::nanoseconds Result();

 private:
  const uint32_t nb_ranks_;
  std::vector<std::unique_ptr<std::chrono::nanoseconds>> duration_;
  std::vector<std::unique_ptr<std::chrono::high_resolution_clock::time_point>> pending_;
  std::vector<std::unique_ptr<bool>> started_;
};

class Timers {
 public:
  explicit Timers(uint32_t nb_ranks) : nb_ranks_(nb_ranks) {}

  std::shared_ptr<Timer> New(const std::string& name);
  std::shared_ptr<Timer> Get(const std::string& name);

  const std::map<std::string, std::shared_ptr<Timer>>& get() const;

 private:
  uint32_t nb_ranks_;
  std::map<std::string, std::shared_ptr<Timer>> timers_;
};

}  // namespace timer
