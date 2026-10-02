// A real llmi::server::Server, bound to an ephemeral loopback port, driven
// by cpp-httplib's own client -- the same transport the server uses (ADR
// 0008) -- over actual HTTP. Everything underneath (a tiny random-weight
// model, a tiny byte-alphabet tokenizer, a minimal chat template) is
// synthetic, the same pattern transformer_test.cpp's make_tiny() and
// tokenizer_test.cpp's tiny_tokenizer() already use, so this suite needs no
// network access and no real model files -- it runs in the same `ctest`
// job as every other unit test.
#include "llmi/server/server.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"
#include "llmi/chat/template.hpp"
#include "llmi/model/model.hpp"
#include "llmi/model/safetensors.hpp"
#include "llmi/model/transformer.hpp"
#include "llmi/tokenizer/tokenizer.hpp"
#include "llmi/util/json.hpp"

using namespace llmi;
using namespace llmi::server;

namespace {

// A byte-alphabet tokenizer (every byte has a token; "a b" and "ab c" also
// merge), 258 entries -- matches tiny_tokenizer() in tests/tokenizer_test.cpp,
// duplicated here per this project's convention that each test file's
// fixtures are self-contained.
std::string tiny_tokenizer_json() {
  std::vector<std::string> alphabet;
  int next = 256;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
    const int cp = printable ? b : next++;
    std::string s;
    if (cp < 0x80) {
      s.push_back(static_cast<char>(cp));
    } else {
      s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    alphabet.push_back(s);
  }
  std::string v;
  int id = 0;
  for (const auto& s : alphabet) {
    const std::string q = s == "\"" ? "\\\"" : s == "\\" ? "\\\\" : s;
    v += (id != 0 ? "," : "") + std::string("\"") + q + "\":" + std::to_string(id);
    ++id;
  }
  v += ",\"ab\":256,\"abc\":257";
  return R"({"normalizer":null,)"
         R"("pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"use_regex":true},)"
         R"("post_processor":null,"decoder":{"type":"ByteLevel"},"added_tokens":[],)"
         R"("model":{"type":"BPE","dropout":null,"unk_token":null,"byte_fallback":false,)"
         R"("vocab":{)" + v + R"(},"merges":["a b","ab c"]}})";
}

// A tiny random-weight Llama-shaped model whose vocab_size matches the
// tokenizer above exactly, so every sampled token id decodes to something.
Transformer make_tiny_transformer() {
  const char* kCfg = R"({
    "architectures": ["LlamaForCausalLM"], "vocab_size": 258, "hidden_size": 16, "intermediate_size": 24,
    "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": true, "eos_token_id": 10
  })";
  auto cfg = parse_config(kCfg);
  if (!cfg) throw std::runtime_error(cfg.error());
  const auto tensors = expected_tensors(*cfg);
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.0F, 0.1F);
  std::string header = "{";
  std::vector<float> data;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    std::uint64_t n = 1;
    std::string shape;
    for (std::size_t d = 0; d < tensors[i].shape.size(); ++d) {
      n *= tensors[i].shape[d];
      shape += (d != 0 ? "," : "") + std::to_string(tensors[i].shape[d]);
    }
    const bool norm = tensors[i].name.find("norm") != std::string::npos;
    const std::uint64_t off = data.size() * 4;
    for (std::uint64_t k = 0; k < n; ++k) data.push_back(norm ? 1.0F + 0.01F * nd(rng) : nd(rng));
    header += (i != 0 ? "," : "") + std::string("\"") + tensors[i].name + "\":{\"dtype\":\"F32\",\"shape\":[" +
              shape + "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + n * 4) + "]}";
  }
  header += "}";
  std::vector<std::byte> bytes(8 + header.size() + data.size() * 4);
  for (int k = 0; k < 8; ++k) {
    bytes[static_cast<std::size_t>(k)] = std::byte((header.size() >> (8U * static_cast<unsigned>(k))) & 0xFFU);
  }
  std::memcpy(bytes.data() + 8, header.data(), header.size());
  std::memcpy(bytes.data() + 8 + header.size(), data.data(), data.size() * 4);
  auto st = SafetensorsFile::parse(bytes);
  if (!st) throw std::runtime_error(st.error());
  auto m = Model::from_parts(*cfg, std::move(st.value()));
  if (!m) throw std::runtime_error(m.error());
  auto tf = Transformer::load(*m);
  if (!tf) throw std::runtime_error(tf.error());
  return std::move(tf.value());
}

Tokenizer make_tiny_tokenizer() {
  auto t = Tokenizer::from_json(tiny_tokenizer_json());
  if (!t) throw std::runtime_error(t.error());
  return std::move(t.value());
}

chat::ChatTemplate make_tiny_chat_template() {
  auto t = chat::ChatTemplate::parse(
      "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
      "{% if add_generation_prompt %}assistant:{% endif %}");
  if (!t) throw std::runtime_error(t.error());
  return std::move(t.value());
}

// Starts a Server on an ephemeral loopback port in a background thread and
// stops it on destruction, so each test gets an independent instance.
class RunningServer {
 public:
  explicit RunningServer(ServerConfig config = {}) {
    config.port = 0;  // let the OS pick a free port
    server_ = std::make_unique<Server>(std::move(config), make_tiny_transformer(), make_tiny_tokenizer(),
                                        make_tiny_chat_template());
    thread_ = std::thread([this] { (void)server_->listen(); });
    // bound_port() becomes valid once bind_to_any_port() returns, inside
    // listen(); poll briefly rather than sleeping a fixed guess.
    for (int i = 0; i < 2000 && server_->bound_port() < 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  ~RunningServer() {
    server_->stop();
    thread_.join();
  }
  RunningServer(const RunningServer&) = delete;
  RunningServer& operator=(const RunningServer&) = delete;

  [[nodiscard]] int port() const { return server_->bound_port(); }

 private:
  std::unique_ptr<Server> server_;
  std::thread thread_;
};

}  // namespace

TEST(ServerIntegration, HealthAndModels) {
  RunningServer rs;
  ASSERT_GT(rs.port(), 0);
  httplib::Client cli("127.0.0.1", rs.port());
  auto health = cli.Get("/health");
  ASSERT_TRUE(health);
  EXPECT_EQ(health->status, 200);
  EXPECT_EQ(health->body, "ok");

  auto models = cli.Get("/v1/models");
  ASSERT_TRUE(models);
  EXPECT_EQ(models->status, 200);
  auto v = json::parse(models->body);
  ASSERT_TRUE(v) << v.error();
  EXPECT_EQ(v->find("data")->items().at(0).find("id")->text(), "llmi");
}

TEST(ServerIntegration, NonStreamingChatCompletion) {
  RunningServer rs;
  httplib::Client cli("127.0.0.1", rs.port());
  auto res = cli.Post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":8})",
                       "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 200);
  auto v = json::parse(res->body);
  ASSERT_TRUE(v) << v.error();
  const auto& choice = v->find("choices")->items().at(0);
  EXPECT_EQ(choice.find("message")->find("role")->text(), "assistant");
  EXPECT_TRUE(choice.find("message")->find("content")->is_string());
  const std::string finish(choice.find("finish_reason")->text());
  EXPECT_TRUE(finish == "stop" || finish == "length");
  const auto completion_tokens = v->find("usage")->find("completion_tokens")->as_u64();
  ASSERT_TRUE(completion_tokens.has_value());
  EXPECT_LE(*completion_tokens, 8U);
  EXPECT_GE(*completion_tokens, 1U);
}

TEST(ServerIntegration, StreamingChatCompletionSendsSseChunksThenDone) {
  RunningServer rs;
  httplib::Client cli("127.0.0.1", rs.port());
  auto res = cli.Post("/v1/chat/completions",
                       R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":8,"stream":true})",
                       "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 200);
  EXPECT_NE(res->body.find("data: "), std::string::npos);
  EXPECT_TRUE(res->body.ends_with("data: [DONE]\n\n"));
  EXPECT_NE(res->body.find("chat.completion.chunk"), std::string::npos);
}

TEST(ServerIntegration, RejectsOversizedBody) {
  ServerConfig config;
  config.max_body_bytes = 64;
  RunningServer rs(config);
  httplib::Client cli("127.0.0.1", rs.port());
  const std::string huge_content(1024, 'x');
  auto res = cli.Post("/v1/chat/completions",
                       R"({"messages":[{"role":"user","content":")" + huge_content + R"("}]})", "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 413);
}

TEST(ServerIntegration, RejectsAtTheConcurrencyLimit) {
  ServerConfig config;
  config.max_concurrent_requests = 0;  // every request is already "over" the limit
  RunningServer rs(config);
  httplib::Client cli("127.0.0.1", rs.port());
  auto res = cli.Post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})",
                       "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 429);
}

TEST(ServerIntegration, RejectsAnInvalidRequestBody) {
  RunningServer rs;
  httplib::Client cli("127.0.0.1", rs.port());
  auto res = cli.Post("/v1/chat/completions", R"({"messages":[]})", "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 400);
  auto v = json::parse(res->body);
  ASSERT_TRUE(v) << v.error();
  EXPECT_TRUE(v->find("error")->find("message")->is_string());
}
