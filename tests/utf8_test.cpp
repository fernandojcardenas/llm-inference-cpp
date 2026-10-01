#include "llmi/util/utf8.hpp"

#include <gtest/gtest.h>

using namespace llmi;

TEST(Utf8, DecodesEveryLength) {
  auto cps = utf8::decode("a\xC3\xA9\xE6\x9D\xB1\xF0\x9F\x99\x82");  // a é 東 🙂
  ASSERT_TRUE(cps);
  EXPECT_EQ(*cps, (std::vector<char32_t>{U'a', 0xE9, 0x6771, 0x1F642}));
}

TEST(Utf8, RejectsMalformedInput) {
  EXPECT_FALSE(utf8::valid("\x80"));              // lone continuation byte
  EXPECT_FALSE(utf8::valid("\xC0\xAF"));          // overlong '/'
  EXPECT_FALSE(utf8::valid("\xE0\x80\xAF"));      // overlong
  EXPECT_FALSE(utf8::valid("\xED\xA0\x80"));      // surrogate U+D800
  EXPECT_FALSE(utf8::valid("\xF4\x90\x80\x80"));  // above U+10FFFF
  EXPECT_FALSE(utf8::valid("\xE6\x9D"));          // truncated
  EXPECT_FALSE(utf8::valid("\xFF"));
  EXPECT_TRUE(utf8::valid(""));
  EXPECT_TRUE(utf8::valid("\xF4\x8F\xBF\xBF"));   // U+10FFFF
}

TEST(Utf8, AppendRoundTrips) {
  for (char32_t cp : {char32_t{0}, char32_t{0x7F}, char32_t{0x80}, char32_t{0x7FF}, char32_t{0x800},
                      char32_t{0xFFFF}, char32_t{0x10000}, char32_t{0x10FFFF}}) {
    std::string s;
    utf8::append(s, cp);
    auto back = utf8::decode(s);
    ASSERT_TRUE(back);
    ASSERT_EQ(back->size(), 1U);
    EXPECT_EQ((*back)[0], cp);
  }
}
