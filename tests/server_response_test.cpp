#include "llmi/server/response.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "llmi/util/json.hpp"

using namespace llmi::server;

namespace {
// Round-trips through this engine's own JSON parser, so these tests check
// the *shape* (valid JSON, the right fields at the right paths) rather than
// byte-exact string matching, which would be brittle against harmless
// reordering.
llmi::json::Value parse(const std::string& s) {
  auto v = llmi::json::parse(s);
  if (!v) throw std::runtime_error(v.error());
  return std::move(v.value());
}
}  // namespace

TEST(ServerResponse, FinishReasonNames) {
  EXPECT_EQ(finish_reason_name(FinishReason::Stop), "stop");
  EXPECT_EQ(finish_reason_name(FinishReason::Length), "length");
}

TEST(ServerResponse, BuildsANonStreamingCompletion) {
  const Usage usage{10, 3};
  const std::string out = build_chat_completion("chatcmpl-1", 1700000000, "llmi", "hi there", FinishReason::Stop,
                                                 usage);
  const auto v = parse(out);
  EXPECT_EQ(v.find("id")->text(), "chatcmpl-1");
  EXPECT_EQ(v.find("object")->text(), "chat.completion");
  EXPECT_EQ(v.find("model")->text(), "llmi");
  const auto& choice = v.find("choices")->items().at(0);
  EXPECT_EQ(choice.find("message")->find("role")->text(), "assistant");
  EXPECT_EQ(choice.find("message")->find("content")->text(), "hi there");
  EXPECT_EQ(choice.find("finish_reason")->text(), "stop");
  EXPECT_EQ(v.find("usage")->find("prompt_tokens")->as_u64(), 10U);
  EXPECT_EQ(v.find("usage")->find("completion_tokens")->as_u64(), 3U);
  EXPECT_EQ(v.find("usage")->find("total_tokens")->as_u64(), 13U);
}

TEST(ServerResponse, EscapesContentThatWouldOtherwiseBreakTheJson) {
  const std::string out =
      build_chat_completion("id", 0, "m", "a \"quote\", a \\backslash and a\nnewline", FinishReason::Length, {});
  const auto v = parse(out);  // throws (via ASSERT_TRUE semantics above) if this isn't valid JSON
  EXPECT_EQ(v.find("choices")->items().at(0).find("message")->find("content")->text(),
            "a \"quote\", a \\backslash and a\nnewline");
}

TEST(ServerResponse, BuildsAContentDeltaChunk) {
  const std::string out = build_chat_completion_chunk("id", 0, "m", "hel", std::nullopt);
  ASSERT_TRUE(out.starts_with("data: "));
  ASSERT_TRUE(out.ends_with("\n\n"));
  const auto v = parse(out.substr(6, out.size() - 6 - 2));
  EXPECT_EQ(v.find("object")->text(), "chat.completion.chunk");
  const auto& choice = v.find("choices")->items().at(0);
  EXPECT_EQ(choice.find("delta")->find("content")->text(), "hel");
  EXPECT_TRUE(choice.find("finish_reason")->is_null());
}

TEST(ServerResponse, BuildsAFinalChunkWithNoContentDelta) {
  const std::string out = build_chat_completion_chunk("id", 0, "m", "", FinishReason::Stop);
  const auto v = parse(out.substr(6, out.size() - 6 - 2));
  const auto& choice = v.find("choices")->items().at(0);
  EXPECT_EQ(choice.find("finish_reason")->text(), "stop");
  EXPECT_EQ(choice.find("delta")->members().size(), 0U);
}

TEST(ServerResponse, SseDoneLine) { EXPECT_EQ(sse_done_line(), "data: [DONE]\n\n"); }
