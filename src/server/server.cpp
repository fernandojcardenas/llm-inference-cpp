#include "llmi/server/server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <string_view>

#include "httplib.h"
#include "llmi/model/sampling.hpp"
#include "llmi/server/response.hpp"
#include "llmi/util/json.hpp"
#include "llmi/util/utf8.hpp"

namespace llmi::server {

namespace {

// The result of one generation run, independent of how it's delivered
// (streamed or not) -- generate() below is the one place both the
// streaming and non-streaming request handlers call into.
struct GenerationResult {
  std::string text;
  std::size_t prompt_tokens = 0;
  std::size_t completion_tokens = 0;
  FinishReason finish = FinishReason::Length;
};

}  // namespace

struct Server::Impl {
  ServerConfig config;
  Transformer transformer;
  Tokenizer tokenizer;
  chat::ChatTemplate chat_template;

  // ADR 0008: kernels::matmul's llmi::util::ThreadPool::shared() cannot be
  // called from more than one thread at a time, so every call into
  // Transformer::forward_cached across every request is serialized here.
  // rng is only ever touched while holding this mutex, so it needs no lock
  // of its own.
  std::mutex generation_mutex;
  std::mt19937_64 rng{std::random_device{}()};

  std::atomic<std::size_t> in_flight{0};
  std::atomic<std::uint64_t> next_id{0};

  httplib::Server svr;
  std::atomic<int> bound_port{-1};

  Impl(ServerConfig c, Transformer t, Tokenizer tok, chat::ChatTemplate ct)
      : config(std::move(c)), transformer(std::move(t)), tokenizer(std::move(tok)), chat_template(std::move(ct)) {}

  // Runs the chat template, encodes the prompt, prefills it, then decodes
  // up to req.max_tokens tokens one at a time, stopping early at the
  // model's own eos_token_ids or any of req.stop's string suffixes.
  // on_piece, if set, is called with each generated token's decoded text
  // as it's produced -- the streaming handler uses it to write an SSE
  // chunk per token; the non-streaming handler passes nullptr and reads
  // the accumulated GenerationResult::text once generation finishes.
  Result<GenerationResult> generate(const ChatCompletionRequest& req,
                                     const std::function<void(std::string_view)>& on_piece) {
    const std::lock_guard<std::mutex> lock(generation_mutex);

    auto rendered = chat_template.render(req.messages, true);
    if (!rendered) return fail("chat template render failed: " + rendered.error());
    std::size_t dropped = 0;
    auto prompt = tokenizer.encode(*rendered, true, &dropped);
    if (!prompt) return fail("tokenization failed: " + prompt.error());

    GenerationResult result;
    result.prompt_tokens = prompt->size();
    if (req.max_tokens == 0) return result;

    auto cache = transformer.new_cache(transformer.config().max_position_embeddings);
    if (!cache) return fail("failed to allocate a KV cache: " + cache.error());
    auto logits = transformer.forward_cached(*prompt, *cache);
    if (!logits) return fail("forward pass failed: " + logits.error());

    const auto& eos_ids = transformer.config().eos_token_ids;
    // Raw bytes decoded but not yet confirmed as complete, valid UTF-8.
    // Tokenizer::decode() documents that "a prefix of a sequence may end
    // inside a UTF-8 character" -- a single token can be one byte of a
    // multi-byte character, decoded on its own it's invalid UTF-8, and
    // JSON-encoding that (unlike llmi-chat's fwrite to a terminal) would
    // produce a response body that isn't valid UTF-8 JSON. Bytes are only
    // ever emitted (appended to result.text / passed to on_piece) once
    // they form a complete, valid prefix; anything left over when
    // generation ends is dropped, the same convention Tokenizer::encode
    // already uses for bytes it has no token for.
    std::string pending_bytes;
    for (std::size_t step = 0; step < req.max_tokens; ++step) {
      const TokenId next = sample(logits->data(), logits->size(), req.sampling, rng);
      ++result.completion_tokens;
      const bool is_eos = std::find(eos_ids.begin(), eos_ids.end(), static_cast<std::int32_t>(next)) != eos_ids.end();
      if (!is_eos) {
        auto piece = tokenizer.decode({next}, true);
        if (piece && !piece->empty()) {
          pending_bytes += *piece;
          std::size_t i = 0;
          std::size_t valid_end = 0;
          while (i < pending_bytes.size() && utf8::next(pending_bytes, i)) valid_end = i;
          if (valid_end > 0) {
            const std::string_view emit(pending_bytes.data(), valid_end);
            result.text += emit;
            if (on_piece) on_piece(emit);
            pending_bytes.erase(0, valid_end);
          }
        }
      }
      if (is_eos) {
        result.finish = FinishReason::Stop;
        break;
      }

      bool stop_matched = false;
      for (const auto& s : req.stop) {
        if (!s.empty() && result.text.size() >= s.size() &&
            result.text.compare(result.text.size() - s.size(), s.size(), s) == 0) {
          result.text.erase(result.text.size() - s.size());
          stop_matched = true;
          break;
        }
      }
      if (stop_matched) {
        result.finish = FinishReason::Stop;
        break;
      }
      if (step + 1 == req.max_tokens) {
        result.finish = FinishReason::Length;
        break;
      }

      auto next_logits = transformer.forward_cached({next}, *cache);
      if (!next_logits) return fail("forward pass failed: " + next_logits.error());
      logits = std::move(next_logits);
    }
    return result;
  }
};

Server::Server(ServerConfig config, Transformer transformer, Tokenizer tokenizer, chat::ChatTemplate chat_template)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(transformer), std::move(tokenizer),
                                    std::move(chat_template))) {
  impl_->svr.set_payload_max_length(impl_->config.max_body_bytes);

  impl_->svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("ok", "text/plain");
  });

  impl_->svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
    std::string out = R"({"object":"list","data":[{"id":)";
    json::append_quoted(out, impl_->config.model_name);
    out += R"(,"object":"model","owned_by":"llmi"}]})";
    res.set_content(out, "application/json");
  });

  impl_->svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
    const std::size_t current = impl_->in_flight.fetch_add(1) + 1;
    const struct InFlightGuard {
      std::atomic<std::size_t>& count;
      ~InFlightGuard() { count.fetch_sub(1); }
    } guard{impl_->in_flight};

    if (current > impl_->config.max_concurrent_requests) {
      res.status = 429;
      res.set_content(
          R"({"error":{"message":"server is at its concurrent request limit","type":"rate_limit_error"}})",
          "application/json");
      return;
    }

    auto parsed = parse_chat_completion_request(req.body, impl_->config.request_limits);
    if (!parsed) {
      res.status = 400;
      std::string out = R"({"error":{"message":)";
      json::append_quoted(out, parsed.error());
      out += R"(,"type":"invalid_request_error"}})";
      res.set_content(out, "application/json");
      return;
    }

    const std::string model_name = parsed->model.empty() ? impl_->config.model_name : parsed->model;
    const std::string id = "chatcmpl-" + std::to_string(impl_->next_id.fetch_add(1));
    const auto created = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());

    if (!parsed->stream) {
      auto gen = impl_->generate(*parsed, nullptr);
      if (!gen) {
        res.status = 500;
        std::string out = R"({"error":{"message":)";
        json::append_quoted(out, gen.error());
        out += R"(,"type":"server_error"}})";
        res.set_content(out, "application/json");
        return;
      }
      const Usage usage{gen->prompt_tokens, gen->completion_tokens};
      res.set_content(build_chat_completion(id, created, model_name, gen->text, gen->finish, usage),
                       "application/json");
      return;
    }

    // Streaming: the content provider below runs the whole generation in
    // one call, writing one SSE chunk per token as on_piece fires, so the
    // generation mutex (held for the duration of Impl::generate) and the
    // HTTP write for that token are both covered by this one invocation --
    // simpler than re-acquiring the lock per token, at the cost of holding
    // it slightly longer than the bare minimum (ADR 0008 only requires that
    // generation itself never overlaps across requests, not that writes
    // stay outside the lock).
    ChatCompletionRequest req_copy = *parsed;
    res.set_chunked_content_provider(
        "text/event-stream", [this, req_copy = std::move(req_copy), id, created, model_name](std::size_t,
                                                                                               httplib::DataSink& sink) {
          bool write_failed = false;
          auto gen = impl_->generate(req_copy, [&](std::string_view piece) {
            if (write_failed) return;
            const std::string chunk = build_chat_completion_chunk(id, created, model_name, piece, std::nullopt);
            if (!sink.write(chunk.data(), chunk.size())) write_failed = true;
          });
          if (!write_failed) {
            const FinishReason finish = gen ? gen->finish : FinishReason::Stop;
            const std::string last = build_chat_completion_chunk(id, created, model_name, "", finish);
            sink.write(last.data(), last.size());
            const auto done = sse_done_line();
            sink.write(done.data(), done.size());
          }
          sink.done();
          return true;
        });
  });
}

Server::~Server() = default;

Result<std::nullptr_t> Server::listen() {
  // bind_to_any_port always binds an ephemeral port (used by port == 0,
  // e.g. tests); a specific requested port needs bind_to_port instead --
  // cpp-httplib's own API splits these into two calls rather than one that
  // takes an explicit-or-zero port.
  int port = 0;
  if (impl_->config.port == 0) {
    port = impl_->svr.bind_to_any_port(impl_->config.host);
    if (port < 0) return fail("failed to bind " + impl_->config.host + " to an ephemeral port");
  } else {
    if (!impl_->svr.bind_to_port(impl_->config.host, impl_->config.port)) {
      return fail("failed to bind " + impl_->config.host + ":" + std::to_string(impl_->config.port));
    }
    port = impl_->config.port;
  }
  impl_->bound_port.store(port);
  impl_->svr.listen_after_bind();
  return nullptr;
}

int Server::bound_port() const { return impl_->bound_port.load(); }

void Server::stop() { impl_->svr.stop(); }

}  // namespace llmi::server
