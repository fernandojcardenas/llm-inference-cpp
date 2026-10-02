#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "llmi/chat/template.hpp"
#include "llmi/model/transformer.hpp"
#include "llmi/server/request.hpp"
#include "llmi/tokenizer/tokenizer.hpp"
#include "llmi/util/result.hpp"

namespace llmi::server {

struct ServerConfig {
  // Loopback by default (ADR 0008): an OpenAI-compatible endpoint with no
  // built-in authentication should not be reachable from the network
  // unless a caller explicitly asks for that by setting a different host.
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string model_name = "llmi";

  // A bounded queue ahead of the single generation slot (ADR 0008): a
  // request beyond this count is rejected with 429 rather than queued
  // without limit.
  std::size_t max_concurrent_requests = 4;
  // Request bodies beyond this size are rejected with 413 before being
  // read into memory.
  std::size_t max_body_bytes = 1U << 20U;  // 1 MiB

  RequestLimits request_limits;
};

// An OpenAI-compatible chat completions server over this engine's own
// Transformer/Tokenizer/ChatTemplate. HTTP accept/parse/response-write is
// concurrent (cpp-httplib's own thread pool); every call into
// Transformer::forward_cached is serialized behind one mutex, because
// kernels::matmul's llmi::util::ThreadPool::shared() cannot be called
// concurrently from more than one thread (ADR 0008).
class Server {
 public:
  Server(ServerConfig config, Transformer transformer, Tokenizer tokenizer, chat::ChatTemplate chat_template);
  ~Server();
  Server(Server&&) = delete;
  Server& operator=(Server&&) = delete;
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  // Binds the configured host/port and serves until stop() is called from
  // another thread (or the process ends). Returns an error if the port
  // cannot be bound.
  Result<std::nullptr_t> listen();

  // The actual bound port -- the same as config().port unless 0 (ephemeral
  // port) was requested, which tests use to avoid picking a fixed port.
  // Valid only after a successful bind (listen() has been called, from
  // another thread, and has reached the point of accepting connections).
  [[nodiscard]] int bound_port() const;

  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace llmi::server
