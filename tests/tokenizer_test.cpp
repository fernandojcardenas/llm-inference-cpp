#include "llmi/tokenizer/tokenizer.hpp"

#include <gtest/gtest.h>

using namespace llmi;

namespace {

const Tokenizer& smollm2() {
  static const Tokenizer t = [] {
    auto r = Tokenizer::load(std::string(LLMI_TESTDATA_DIR) + "/smollm2-135m/tokenizer.json");
    if (!r) throw std::runtime_error(r.error());
    return std::move(r.value());
  }();
  return t;
}

std::vector<TokenId> enc(const std::string& s) {
  auto r = smollm2().encode(s);
  if (!r) throw std::runtime_error(r.error());
  return r.value();
}

}  // namespace

// Expected ids from Hugging Face tokenizers 0.23.2 on the same tokenizer.json.
TEST(Tokenizer, MatchesHuggingFaceOnKnownStrings) {
  EXPECT_EQ(enc("Hello, world!"), (std::vector<TokenId>{19556, 28, 905, 17}));
  EXPECT_EQ(enc("The year 2026 had 365 days."),
            (std::vector<TokenId>{504, 713, 216, 34, 32, 34, 38, 761, 216, 35, 38, 37, 2009, 30}));
  EXPECT_EQ(enc("I'm here, they're not  "), (std::vector<TokenId>{57, 5248, 1535, 28, 502, 2316, 441, 256}));
  EXPECT_EQ(enc("<|im_start|>user\nHi<|im_end|>"), (std::vector<TokenId>{1, 4093, 198, 26843, 2}));
  EXPECT_EQ(enc("na\xC3\xAFve caf\xC3\xA9 \xE6\x9D\xB1\xE4\xBA\xAC \xF0\x9F\x99\x82"),
            (std::vector<TokenId>{3546, 46494, 37366, 17097, 247, 126, 16736, 122, 47526}));
  EXPECT_EQ(enc("   leading and trailing   "), (std::vector<TokenId>{256, 2899, 284, 35079, 333}));
  EXPECT_EQ(enc("x\n\n\ny"), (std::vector<TokenId>{104, 1116, 198, 105}));
  EXPECT_EQ(enc("int main() { return 0; }"), (std::vector<TokenId>{591, 1085, 1000, 1662, 1003, 216, 32, 43, 4029}));
  EXPECT_TRUE(enc("").empty());
}

TEST(Tokenizer, DecodeInvertsEncode) {
  for (const std::string& s : std::vector<std::string>{"Hello, world!", "  tabs\tand\r\nnewlines \n", "\xF0\x9F\x99\x82\xF0\x9F\x99\x82",
                              "<|im_start|>system\n<|im_end|>", "123456789", std::string("nul\0byte", 8)}) {
    auto back = smollm2().decode(enc(s));
    ASSERT_TRUE(back) << back.error();
    EXPECT_EQ(back.value(), s);
  }
}

// SmolLM2's vocabulary has no token for six ASCII control characters; the
// reference implementation drops them silently. The engine matches it, and
// reports how many bytes were dropped.
TEST(Tokenizer, DropsBytesWithoutATokenLikeTheReference) {
  EXPECT_EQ(smollm2().missing_bytes(),
            (std::vector<std::uint8_t>{4, 6, 19, 20, 22, 29, 192, 193, 241, 242, 245, 246, 247, 248, 249, 250, 251,
                                       252, 253, 254, 255}));
  std::size_t dropped = 99;
  auto ids = smollm2().encode(std::string("a\x04" "b\0c", 5), true, &dropped);
  ASSERT_TRUE(ids);
  EXPECT_EQ(ids.value(), (std::vector<TokenId>{81, 82, 190, 83}));
  EXPECT_EQ(dropped, 1U);
  ids = smollm2().encode("\x04\x06\x13\x14\x16\x1d", true, &dropped);
  EXPECT_TRUE(ids->empty());
  EXPECT_EQ(dropped, 6U);
  ids = smollm2().encode("x \x1d y", true, &dropped);  // cached words report drops too
  EXPECT_EQ(ids.value(), (std::vector<TokenId>{104, 216, 329}));
  EXPECT_EQ(dropped, 1U);
  ids = smollm2().encode("x \x1d y", true, &dropped);
  EXPECT_EQ(dropped, 1U);
  ids = smollm2().encode("clean text", true, &dropped);
  EXPECT_EQ(dropped, 0U);
}

TEST(Tokenizer, CanTreatSpecialTokensAsText) {
  auto r = smollm2().encode("<|im_end|>", /*parse_special=*/false);
  ASSERT_TRUE(r);
  EXPECT_GT(r->size(), 1U);
  for (TokenId id : r.value()) EXPECT_FALSE(smollm2().is_special(id));
  auto back = smollm2().decode(r.value());
  EXPECT_EQ(back.value(), "<|im_end|>");
}

TEST(Tokenizer, SkipsSpecialTokensWhenAsked) {
  auto back = smollm2().decode(enc("<|im_start|>Hi<|im_end|>"), /*skip_special=*/true);
  ASSERT_TRUE(back);
  EXPECT_EQ(back.value(), "Hi");
}

TEST(Tokenizer, RejectsInvalidInput) {
  EXPECT_FALSE(smollm2().encode("bad \xC0\xAF utf8"));
  EXPECT_FALSE(smollm2().decode({-1}));
  EXPECT_FALSE(smollm2().decode({static_cast<TokenId>(smollm2().vocab_size())}));
}

TEST(Tokenizer, LooksUpTokens) {
  EXPECT_EQ(smollm2().vocab_size(), 49152U);
  EXPECT_EQ(smollm2().id("Hello"), 19556);
  EXPECT_EQ(smollm2().token(19556), "Hello");
  EXPECT_EQ(smollm2().id("not a token at all"), -1);
  EXPECT_TRUE(smollm2().is_special(0));
  EXPECT_FALSE(smollm2().is_special(19556));
}

namespace {

// A tiny tokenizer.json: byte alphabet plus merges "a b" -> "ab", "ab c" -> "abc".
std::string tiny_tokenizer(const std::string& extra_model = "", const std::string& pre = "") {
  std::string vocab;
  // The 188 printable bytes map to themselves; the rest to U+0100.. in order.
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
    std::string q = s == "\"" ? "\\\"" : s == "\\" ? "\\\\" : s;
    v += (id != 0 ? "," : "") + std::string("\"") + q + "\":" + std::to_string(id);
    ++id;
  }
  v += ",\"ab\":256,\"abc\":257";
  const std::string pre_tok = pre.empty() ? R"({"type":"ByteLevel","add_prefix_space":false,"use_regex":true})" : pre;
  return R"({"normalizer":null,"pre_tokenizer":)" + pre_tok +
         R"(,"post_processor":null,"decoder":{"type":"ByteLevel"},"added_tokens":[],)"
         R"("model":{"type":"BPE","dropout":null,"unk_token":null,"byte_fallback":false)" + extra_model +
         R"(,"vocab":{)" + v + R"(},"merges":["a b","ab c"]}})";
}

}  // namespace

TEST(Tokenizer, AppliesMergesByRank) {
  auto t = Tokenizer::from_json(tiny_tokenizer());
  ASSERT_TRUE(t) << t.error();
  EXPECT_EQ(t->encode("abc").value(), (std::vector<TokenId>{257}));
  EXPECT_EQ(t->encode("abab").value(), (std::vector<TokenId>{256, 256}));
  EXPECT_EQ(t->encode("acb").value(), (std::vector<TokenId>{t->id("a"), t->id("c"), t->id("b")}));
}

TEST(Tokenizer, RejectsUnsupportedFeatures) {
  EXPECT_FALSE(Tokenizer::from_json(tiny_tokenizer(R"(,"byte_fallback":true)")));  // duplicate key
  EXPECT_FALSE(Tokenizer::from_json(tiny_tokenizer("", R"({"type":"Metaspace"})")));
  EXPECT_FALSE(Tokenizer::from_json(tiny_tokenizer("", R"({"type":"ByteLevel","add_prefix_space":true})")));
  EXPECT_FALSE(Tokenizer::from_json(tiny_tokenizer("", R"({"type":"ByteLevel","use_regex":false})")));
  std::string with_norm = tiny_tokenizer();
  with_norm.replace(with_norm.find("\"normalizer\":null"), 17, R"("normalizer":{"type":"NFC"})");
  EXPECT_FALSE(Tokenizer::from_json(with_norm));
  std::string bad_merge = tiny_tokenizer();
  bad_merge.replace(bad_merge.find("\"ab c\""), 6, "\"ab zz\"");
  EXPECT_FALSE(Tokenizer::from_json(bad_merge));
}

TEST(Tokenizer, ExposesTheUnicodeTablesVersion) {
  EXPECT_TRUE(unicode::is_letter(U'a'));
  EXPECT_TRUE(unicode::is_letter(0x6771));  // 東
  EXPECT_TRUE(unicode::is_number(U'7'));
  EXPECT_TRUE(unicode::is_number(0x2163));  // Ⅳ (Nl)
  EXPECT_FALSE(unicode::is_letter(U'7'));
  EXPECT_TRUE(unicode::is_whitespace(0x3000));
  EXPECT_FALSE(unicode::is_whitespace(0x200B));  // zero-width space is not White_Space
  EXPECT_STRNE(unicode::version(), "");
}
