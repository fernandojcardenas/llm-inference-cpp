#include "llmi/util/json.hpp"

#include <gtest/gtest.h>

using namespace llmi;

TEST(Json, ParsesNestedValues) {
  auto v = json::parse(R"( {"a": [1, -2.5e3, true, false, null], "b": {"c": "d"}} )");
  ASSERT_TRUE(v) << v.error();
  ASSERT_TRUE(v->is_object());
  const auto* a = v->find("a");
  ASSERT_NE(a, nullptr);
  ASSERT_EQ(a->items().size(), 5U);
  EXPECT_EQ(a->items()[0].as_u64(), 1U);
  EXPECT_EQ(a->items()[1].as_double(), -2500.0);
  EXPECT_FALSE(a->items()[1].as_u64());
  EXPECT_TRUE(a->items()[2].as_bool());
  EXPECT_TRUE(a->items()[4].is_null());
  EXPECT_EQ(v->find("b")->find("c")->text(), "d");
  EXPECT_EQ(v->find("missing"), nullptr);
}

TEST(Json, ReadsLargeIntegersExactly) {
  auto v = json::parse("[18446744073709551615, 9007199254740993, 18446744073709551616, -1]");
  ASSERT_TRUE(v);
  EXPECT_EQ(v->items()[0].as_u64(), 18446744073709551615ULL);
  EXPECT_EQ(v->items()[1].as_u64(), 9007199254740993ULL);  // not representable as a double
  EXPECT_FALSE(v->items()[2].as_u64());                     // overflow
  EXPECT_FALSE(v->items()[3].as_u64());                     // negative
  EXPECT_EQ(v->items()[3].as_i64(), -1);
}

TEST(Json, DecodesEscapes) {
  auto v = json::parse(R"("a\"b\\c\/\n\u00e9\ud83d\ude42")");
  ASSERT_TRUE(v) << v.error();
  EXPECT_EQ(v->text(), "a\"b\\c/\n\xC3\xA9\xF0\x9F\x99\x82");
}

TEST(Json, RejectsMalformedInput) {
  for (const char* bad : {"", "{", "[1,]", "{\"a\":1,}", "{\"a\" 1}", "01", "1.", "-", "+1", "1e", "tru", "nul",
                          "\"unterminated", "\"\\x\"", "\"\\ud800\"", "\"\\udc00\"", "\"\\ud800\\u0041\"",
                          "[1] 2", "{\"a\":1,\"a\":2}", "\"tab\there\"", "\"\xC0\xAF\"", "NaN", "Infinity"}) {
    EXPECT_FALSE(json::parse(bad)) << bad;
  }
}

TEST(Json, LimitsNestingDepth) {
  std::string deep(100, '[');
  deep += std::string(100, ']');
  EXPECT_FALSE(json::parse(deep));
  json::Limits l;
  l.max_depth = 200;
  EXPECT_TRUE(json::parse(deep, l));
}

TEST(Json, LimitsNodeCount) {
  json::Limits l;
  l.max_nodes = 3;
  EXPECT_TRUE(json::parse("[1,2]", l));
  EXPECT_FALSE(json::parse("[1,2,3]", l));
}

TEST(Json, QuotesStrings) {
  std::string out;
  json::append_quoted(out, std::string("a\"\\\n\x01", 5));
  EXPECT_EQ(out, R"("a\"\\\n\u0001")");
  auto back = json::parse(out);
  ASSERT_TRUE(back);
  EXPECT_EQ(back->text(), std::string("a\"\\\n\x01", 5));
}
