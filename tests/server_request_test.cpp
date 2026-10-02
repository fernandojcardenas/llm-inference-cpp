#include "llmi/server/request.hpp"

#include <gtest/gtest.h>

using namespace llmi::server;

namespace {
RequestLimits default_limits() { return {}; }
}  // namespace

TEST(ServerRequest, ParsesAMinimalRequest) {
  auto r = parse_chat_completion_request(R"({"messages":[{"role":"user","content":"hi"}]})", default_limits());
  ASSERT_TRUE(r) << r.error();
  EXPECT_EQ(r->messages.size(), 1U);
  EXPECT_EQ(r->messages[0].role, "user");
  EXPECT_EQ(r->messages[0].content, "hi");
  EXPECT_EQ(r->max_tokens, default_limits().default_max_tokens);
  EXPECT_FALSE(r->stream);
  EXPECT_TRUE(r->stop.empty());
}

TEST(ServerRequest, ParsesEveryFieldThisEngineUses) {
  auto r = parse_chat_completion_request(
      R"({"model":"llmi","messages":[{"role":"system","content":"be terse"},
          {"role":"user","content":"hi"}],"max_tokens":32,"temperature":0.7,
          "top_p":0.9,"top_k":40,"stream":true,"stop":["\n\n","END"]})",
      default_limits());
  ASSERT_TRUE(r) << r.error();
  EXPECT_EQ(r->model, "llmi");
  EXPECT_EQ(r->messages.size(), 2U);
  EXPECT_EQ(r->max_tokens, 32U);
  EXPECT_FLOAT_EQ(r->sampling.temperature, 0.7F);
  EXPECT_FLOAT_EQ(r->sampling.top_p, 0.9F);
  EXPECT_EQ(r->sampling.top_k, 40U);
  EXPECT_TRUE(r->stream);
  EXPECT_EQ(r->stop, (std::vector<std::string>{"\n\n", "END"}));
}

TEST(ServerRequest, AcceptsASingleStringStop) {
  auto r = parse_chat_completion_request(R"({"messages":[{"role":"user","content":"x"}],"stop":"END"})",
                                          default_limits());
  ASSERT_TRUE(r) << r.error();
  EXPECT_EQ(r->stop, (std::vector<std::string>{"END"}));
}

TEST(ServerRequest, RejectsInvalidJson) {
  EXPECT_FALSE(parse_chat_completion_request("not json", default_limits()));
  EXPECT_FALSE(parse_chat_completion_request("[]", default_limits()));  // top level must be an object
}

TEST(ServerRequest, RejectsMissingOrEmptyMessages) {
  EXPECT_FALSE(parse_chat_completion_request(R"({})", default_limits()));
  EXPECT_FALSE(parse_chat_completion_request(R"({"messages":[]})", default_limits()));
  EXPECT_FALSE(parse_chat_completion_request(R"({"messages":"nope"})", default_limits()));
}

TEST(ServerRequest, RejectsAnUnknownRole) {
  EXPECT_FALSE(
      parse_chat_completion_request(R"({"messages":[{"role":"tool","content":"x"}]})", default_limits()));
}

TEST(ServerRequest, RejectsNonStringContent) {
  EXPECT_FALSE(
      parse_chat_completion_request(R"({"messages":[{"role":"user","content":5}]})", default_limits()));
}

TEST(ServerRequest, RejectsMaxTokensAboveTheServersCeiling) {
  RequestLimits limits = default_limits();
  limits.max_tokens_ceiling = 100;
  EXPECT_TRUE(parse_chat_completion_request(R"({"messages":[{"role":"user","content":"x"}],"max_tokens":100})",
                                             limits));
  EXPECT_FALSE(parse_chat_completion_request(R"({"messages":[{"role":"user","content":"x"}],"max_tokens":101})",
                                              limits));
}

TEST(ServerRequest, RejectsTooManyMessages) {
  RequestLimits limits = default_limits();
  limits.max_messages = 1;
  EXPECT_FALSE(parse_chat_completion_request(
      R"({"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]})", limits));
}

TEST(ServerRequest, RejectsOversizedMessageContent) {
  RequestLimits limits = default_limits();
  limits.max_message_bytes = 4;
  EXPECT_FALSE(
      parse_chat_completion_request(R"({"messages":[{"role":"user","content":"too long"}]})", limits));
}

TEST(ServerRequest, RejectsOutOfRangeSamplingParameters) {
  EXPECT_FALSE(parse_chat_completion_request(
      R"({"messages":[{"role":"user","content":"x"}],"temperature":-1})", default_limits()));
  EXPECT_FALSE(parse_chat_completion_request(
      R"({"messages":[{"role":"user","content":"x"}],"temperature":2.1})", default_limits()));
  EXPECT_FALSE(
      parse_chat_completion_request(R"({"messages":[{"role":"user","content":"x"}],"top_p":1.1})", default_limits()));
  EXPECT_FALSE(
      parse_chat_completion_request(R"({"messages":[{"role":"user","content":"x"}],"top_p":-0.1})", default_limits()));
}

TEST(ServerRequest, RejectsTooManyStopSequences) {
  RequestLimits limits = default_limits();
  limits.max_stop_sequences = 1;
  EXPECT_FALSE(
      parse_chat_completion_request(R"({"messages":[{"role":"user","content":"x"}],"stop":["a","b"]})", limits));
}

TEST(ServerRequest, IgnoresUnknownFields) {
  auto r = parse_chat_completion_request(
      R"({"messages":[{"role":"user","content":"x"}],"n":3,"logprobs":true,"user":"abc"})", default_limits());
  EXPECT_TRUE(r) << r.error();
}
