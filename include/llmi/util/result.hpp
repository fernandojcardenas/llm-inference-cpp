#pragma once

#include <string>
#include <utility>
#include <variant>

namespace llmi {

// Why an operation failed. Every loader and parser in this project returns
// errors as values: untrusted input must never abort the process.
struct Error {
  std::string message;
};

// A value or an error (std::expected is C++23; this project targets C++20).
template <typename T>
class Result {
 public:
  Result(T value) : v_(std::move(value)) {}            // NOLINT(google-explicit-constructor)
  Result(Error error) : v_(std::move(error)) {}        // NOLINT(google-explicit-constructor)

  [[nodiscard]] bool ok() const { return std::holds_alternative<T>(v_); }
  explicit operator bool() const { return ok(); }

  [[nodiscard]] T& value() & { return std::get<T>(v_); }
  [[nodiscard]] const T& value() const& { return std::get<T>(v_); }
  [[nodiscard]] T&& value() && { return std::get<T>(std::move(v_)); }
  T* operator->() { return &value(); }
  const T* operator->() const { return &value(); }
  T& operator*() & { return value(); }
  const T& operator*() const& { return value(); }

  [[nodiscard]] const std::string& error() const { return std::get<Error>(v_).message; }

 private:
  std::variant<T, Error> v_;
};

inline Error fail(std::string message) { return Error{std::move(message)}; }

}  // namespace llmi
