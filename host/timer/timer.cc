#include "timer.h"

#include <cassert>
#include <chrono>

namespace timer {

Timer::Timer(uint32_t nb_ranks) : nb_ranks_(nb_ranks) {
  for (uint32_t i = 0; i < nb_ranks; ++i) {
    duration_.emplace_back(
        std::make_unique<std::chrono::nanoseconds>(std::chrono::nanoseconds(0)));
    pending_.emplace_back(
        std::make_unique<std::chrono::high_resolution_clock::time_point>());
    started_.emplace_back(std::make_unique<bool>(false));
  }
}

void Timer::Start(uint32_t id) {
  assert(id < pending_.size());
  *pending_[id] = std::chrono::high_resolution_clock::now();
  *started_[id] = true;
}

void Timer::Stop(uint32_t id) {
  assert(id < pending_.size());
  assert(started_[id]);
  auto duration = std::chrono::high_resolution_clock::now() - *pending_[id];
  *duration_[id] += std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  *started_[id] = false;
}

std::chrono::nanoseconds Timer::Result() {

  std::chrono::nanoseconds time = std::chrono::nanoseconds(0);
  for(uint32_t i = 0; i < nb_ranks_; ++i)
    time += *duration_[i];

  return time;
}

std::shared_ptr<Timer> Timers::New(const std::string& name) {
  assert(timers_.find(name) == timers_.end());
  auto timer = std::make_shared<Timer>(nb_ranks_);
  timers_[name] = timer;
  return timer;
}

std::shared_ptr<Timer> Timers::Get(const std::string& name) {
  auto it = timers_.find(name);
  assert(it != timers_.end());
  return it->second;
}

const std::map<std::string, std::shared_ptr<Timer>>& Timers::get() const {
  return timers_;
}

}
