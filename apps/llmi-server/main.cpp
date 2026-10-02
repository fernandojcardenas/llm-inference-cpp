// llmi-server: an OpenAI-compatible chat completions HTTP server.
//
//   llmi-server MODEL_DIR [--host HOST] [--port N] [--quant f32|q8_0|q4_0]
//               [--max-concurrent N] [--max-tokens N] [--max-body-bytes N]
//
// Binds 127.0.0.1 by default (ADR 0008): pass --host to bind anything else.
// MODEL_DIR needs config.json, model.safetensors, tokenizer.json and a
// tokenizer_config.json with a chat_template, the same requirement
// llmi-chat has.
//
// Every request's actual token generation is serialized behind one mutex
// (ADR 0008: llmi::util::ThreadPool::shared() cannot be called concurrently
// from more than one thread), so throughput under concurrent load is bounded
// by that, not by this process's thread count.

#include <cstdio>
#include <string>

#include "llmi/chat/template.hpp"
#include "llmi/model/model.hpp"
#include "llmi/model/quant.hpp"
#include "llmi/model/transformer.hpp"
#include "llmi/server/server.hpp"
#include "llmi/tokenizer/tokenizer.hpp"

namespace {

struct Args {
  std::string model_dir;
  llmi::server::ServerConfig server;
  llmi::quant::Type quant = llmi::quant::Type::F32;
};

bool parse_quant(const std::string& s, llmi::quant::Type* out) {
  if (s == "f32") *out = llmi::quant::Type::F32;
  else if (s == "q8_0") *out = llmi::quant::Type::Q8_0;
  else if (s == "q4_0") *out = llmi::quant::Type::Q4_0;
  else return false;
  return true;
}

bool parse_args(int argc, char** argv, Args* a) {
  if (argc < 2) return false;
  a->model_dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--host" && has_value) a->server.host = argv[++i];
    else if (arg == "--port" && has_value) a->server.port = std::stoi(argv[++i]);
    else if (arg == "--quant" && has_value) {
      if (!parse_quant(argv[++i], &a->quant)) return false;
    } else if (arg == "--max-concurrent" && has_value) {
      a->server.max_concurrent_requests = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else if (arg == "--max-tokens" && has_value) {
      a->server.request_limits.max_tokens_ceiling = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else if (arg == "--max-body-bytes" && has_value) {
      a->server.max_body_bytes = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, &args)) {
    std::fprintf(stderr,
                 "usage: %s MODEL_DIR [--host HOST] [--port N] [--quant f32|q8_0|q4_0] "
                 "[--max-concurrent N] [--max-tokens N] [--max-body-bytes N]\n",
                 argv[0]);
    return 2;
  }

  if (args.server.host != "127.0.0.1" && args.server.host != "localhost") {
    std::fprintf(stderr,
                 "warning: binding %s is not loopback -- this server has no built-in authentication\n",
                 args.server.host.c_str());
  }

  auto model = llmi::Model::load(args.model_dir);
  if (!model) {
    std::fprintf(stderr, "error: %s\n", model.error().c_str());
    return 1;
  }
  auto transformer = llmi::Transformer::load(model.value(), args.quant);
  if (!transformer) {
    std::fprintf(stderr, "error: %s\n", transformer.error().c_str());
    return 1;
  }
  auto tokenizer = llmi::Tokenizer::load(args.model_dir + "/tokenizer.json");
  if (!tokenizer) {
    std::fprintf(stderr, "error: %s\n", tokenizer.error().c_str());
    return 1;
  }
  auto tpl = llmi::chat::ChatTemplate::load(args.model_dir + "/tokenizer_config.json");
  if (!tpl) {
    std::fprintf(stderr, "error: %s (this model may not have a chat template)\n", tpl.error().c_str());
    return 1;
  }

  llmi::server::Server server(args.server, std::move(transformer.value()), std::move(tokenizer.value()),
                               std::move(tpl.value()));
  std::fprintf(stderr, "llmi-server: listening on %s:%d\n", args.server.host.c_str(), args.server.port);
  auto result = server.listen();
  if (!result) {
    std::fprintf(stderr, "error: %s\n", result.error().c_str());
    return 1;
  }
  return 0;
}
