#include "llmi/chat/template.hpp"

#include <gtest/gtest.h>

using namespace llmi::chat;

namespace {
std::string testdata(const std::string& rel) { return std::string(LLMI_TESTDATA_DIR) + "/" + rel; }
}  // namespace

TEST(ChatTemplate, LoadsFromBothModelsTokenizerConfig) {
  auto qwen = ChatTemplate::load(testdata("qwen2.5-0.5b-instruct/tokenizer_config.json"));
  auto smol = ChatTemplate::load(testdata("smollm2-135m-instruct/tokenizer_config.json"));
  ASSERT_TRUE(qwen) << qwen.error();
  ASSERT_TRUE(smol) << smol.error();
}

TEST(ChatTemplate, LoadFailsWithoutAChatTemplateField) {
  // The base (non-instruct) SmolLM2-135M's tokenizer config has no chat_template.
  auto cfg = ChatTemplate::load(testdata("smollm2-135m/config.json"));
  EXPECT_FALSE(cfg);
}

// --------------------------------------------------------- Qwen2.5 (ChatML with a default system prompt)

class QwenTemplate : public ::testing::Test {
 protected:
  void SetUp() override {
    auto t = ChatTemplate::load(testdata("qwen2.5-0.5b-instruct/tokenizer_config.json"));
    ASSERT_TRUE(t) << t.error();
    tpl = std::move(t.value());
  }
  ChatTemplate tpl;
};

TEST_F(QwenTemplate, SingleUserTurnGetsDefaultSystemPrompt) {
  auto out = tpl.render({{"user", "What is 12 times 7?"}}, true);
  ASSERT_TRUE(out);
  EXPECT_EQ(*out,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nWhat is 12 times 7?<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST_F(QwenTemplate, ExplicitSystemMessageReplacesTheDefault) {
  auto out = tpl.render({{"system", "You are terse."}, {"user", "Hi"}}, true);
  ASSERT_TRUE(out);
  EXPECT_EQ(*out,
            "<|im_start|>system\nYou are terse.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST_F(QwenTemplate, MultiTurnHistory) {
  auto out = tpl.render({{"system", "You are terse."}, {"user", "Hi"}, {"assistant", "Hello."}, {"user", "Bye"}}, true);
  ASSERT_TRUE(out);
  EXPECT_EQ(*out,
            "<|im_start|>system\nYou are terse.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\nHello.<|im_end|>\n"
            "<|im_start|>user\nBye<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST_F(QwenTemplate, WithoutGenerationPromptStopsAfterLastMessage) {
  auto out = tpl.render({{"user", "Hi"}, {"assistant", "Hello."}}, false);
  ASSERT_TRUE(out);
  EXPECT_TRUE(out->ends_with("<|im_end|>\n"));
  EXPECT_EQ(out->find("<|im_start|>assistant\n", out->size() - 1), std::string::npos);
}

// ------------------------------------------------------------- SmolLM2 (inserts its own default system message)

TEST(ChatTemplateSmolLM2, SingleUserTurnGetsDefaultSystemPrompt) {
  auto t = ChatTemplate::load(std::string(LLMI_TESTDATA_DIR) + "/smollm2-135m-instruct/tokenizer_config.json");
  ASSERT_TRUE(t) << t.error();
  auto out = t->render({{"user", "Hi"}}, true);
  ASSERT_TRUE(out);
  EXPECT_EQ(*out,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST(ChatTemplateSmolLM2, ExplicitSystemMessageIsNotDuplicated) {
  auto t = ChatTemplate::load(std::string(LLMI_TESTDATA_DIR) + "/smollm2-135m-instruct/tokenizer_config.json");
  ASSERT_TRUE(t) << t.error();
  auto out = t->render({{"system", "Be brief."}, {"user", "Hi"}}, true);
  ASSERT_TRUE(out);
  EXPECT_EQ(*out,
            "<|im_start|>system\nBe brief.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// ------------------------------------------------------------- engine coverage (not tied to a specific model)

TEST(ChatTemplate, ForIfAndWhitespaceControl) {
  auto t = ChatTemplate::parse(
      "{%- for m in messages -%}\n"
      "[{{ m.role }}:{{ m.content }}]"
      "{%- if not loop.last %}, {% endif -%}\n"
      "{%- endfor -%}");
  ASSERT_TRUE(t) << t.error();
  auto out = t->render({{"user", "a"}, {"assistant", "b"}, {"user", "c"}}, false);
  ASSERT_TRUE(out);
  EXPECT_EQ(*out, "[user:a], [assistant:b], [user:c]");
}

TEST(ChatTemplate, StringConcatenationAndComparison) {
  auto t = ChatTemplate::parse("{{ 'role=' + messages[0]['role'] }}{% if messages[0]['role'] == 'system' %} (system){% endif %}");
  ASSERT_TRUE(t) << t.error();
  auto out = t->render({{"system", "x"}}, false);
  ASSERT_TRUE(out);
  EXPECT_EQ(*out, "role=system (system)");
}

TEST(ChatTemplate, DefaultConstructedFailsToRender) {
  ChatTemplate t;
  EXPECT_FALSE(t.render({{"user", "hi"}}, true));
}

TEST(ChatTemplate, RejectsMalformedSyntax) {
  EXPECT_FALSE(ChatTemplate::parse("{% if true %}no endif"));
  EXPECT_FALSE(ChatTemplate::parse("{{ 1 + }}"));
  EXPECT_FALSE(ChatTemplate::parse("{% bogus_tag %}"));
}
