#include "llmi/model/safetensors.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

using namespace llmi;

namespace {

// A safetensors file: 8-byte little-endian header length, header, data.
std::vector<std::byte> make_file(const std::string& header, std::size_t data_bytes, std::uint64_t claimed_len = ~0ULL) {
  const std::uint64_t len = claimed_len == ~0ULL ? header.size() : claimed_len;
  std::vector<std::byte> out(8 + header.size() + data_bytes);
  for (int k = 0; k < 8; ++k) out[static_cast<std::size_t>(k)] = std::byte((len >> (8U * static_cast<unsigned>(k))) & 0xFFU);
  std::memcpy(out.data() + 8, header.data(), header.size());
  for (std::size_t i = 0; i < data_bytes; ++i) out[8 + header.size() + i] = std::byte(i & 0xFFU);
  return out;
}

Result<SafetensorsFile> parse(const std::vector<std::byte>& f) { return SafetensorsFile::parse(f); }

}  // namespace

TEST(Safetensors, ParsesAValidFile) {
  const auto f = make_file(
      R"({"__metadata__":{"format":"pt"},"b":{"dtype":"BF16","shape":[2,3],"data_offsets":[16,28]},)"
      R"("a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})",
      28);
  auto st = parse(f);
  ASSERT_TRUE(st) << st.error();
  ASSERT_EQ(st->tensors().size(), 2U);
  EXPECT_EQ(st->tensors()[0].name, "a");  // sorted by name
  EXPECT_EQ(st->metadata().at("format"), "pt");
  const TensorInfo* b = st->find("b");
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->dtype, DType::BF16);
  EXPECT_EQ(b->numel(), 6U);
  EXPECT_EQ(st->data(*b).size(), 12U);
  EXPECT_EQ(st->data(*b)[0], std::byte{16});
  EXPECT_EQ(st->find("c"), nullptr);
}

TEST(Safetensors, AcceptsScalarsAndEmptyTensors) {
  const auto f = make_file(
      R"({"s":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"e":{"dtype":"F32","shape":[0,5],"data_offsets":[4,4]}})", 4);
  auto st = parse(f);
  ASSERT_TRUE(st) << st.error();
  EXPECT_EQ(st->find("s")->numel(), 1U);
  EXPECT_EQ(st->find("e")->numel(), 0U);
}

TEST(Safetensors, RejectsBadHeaderLengths) {
  EXPECT_FALSE(SafetensorsFile::parse(std::vector<std::byte>(7)));
  EXPECT_FALSE(parse(make_file("{}", 0, 3)));                  // header length past the end
  EXPECT_FALSE(parse(make_file("{}", 0, 1ULL << 40U)));       // absurd length
  EXPECT_FALSE(parse(make_file("{}", 0, ~0ULL - 1)));         // would overflow 8 + len
  EXPECT_TRUE(parse(make_file("{}", 0)));
}

TEST(Safetensors, RejectsMalformedEntries) {
  const char* bad[] = {
      R"([])",                                                                           // not an object
      R"({"a":1})",                                                                      // entry not an object
      R"({"a":{"dtype":"F32","shape":[1]}})",                                            // missing offsets
      R"({"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,4]}})",                        // unknown dtype
      R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})",                      // negative dim
      R"({"a":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}})",                     // fractional dim
      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}})",                         // one offset
      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})",                       // begin > end
      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1}})",             // unknown field
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})",                       // size mismatch
      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}})",                       // past the data
      R"({"__metadata__":{"k":1},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",  // non-string metadata
      R"({"a":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,4]}})",       // rank > 8
      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
  };
  for (const char* h : bad) EXPECT_FALSE(parse(make_file(h, 4))) << h;
}

TEST(Safetensors, RejectsShapesThatOverflow) {
  // 2^32 * 2^32 * 4 bytes wraps to 0 in 64 bits; it must not match an empty range.
  EXPECT_FALSE(parse(make_file(R"({"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}})", 0)));
  EXPECT_FALSE(parse(make_file(R"({"a":{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,4]}})", 4)));
}

TEST(Safetensors, RequiresTheDataSectionToBeCoveredExactly) {
  // overlap
  EXPECT_FALSE(parse(make_file(
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", 8)));
  // hole
  EXPECT_FALSE(parse(make_file(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", 8)));
  // trailing bytes
  EXPECT_FALSE(parse(make_file(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", 6)));
}

TEST(Safetensors, ConvertsFloatFormats) {
  // F32 1.5, BF16 -2.0 (0xC000), F16 0.333 (0x3555), F16 subnormal (0x0001), F16 inf (0x7C00)
  const std::vector<std::byte> f32 = {std::byte{0x00}, std::byte{0x00}, std::byte{0xC0}, std::byte{0x3F}};
  EXPECT_EQ(read_float(f32, DType::F32, 0), 1.5F);
  const std::vector<std::byte> bf16 = {std::byte{0x00}, std::byte{0xC0}};
  EXPECT_EQ(read_float(bf16, DType::BF16, 0), -2.0F);
  const std::vector<std::byte> f16 = {std::byte{0x55}, std::byte{0x35}, std::byte{0x01}, std::byte{0x00},
                                      std::byte{0x00}, std::byte{0x7C}};
  EXPECT_NEAR(read_float(f16, DType::F16, 0), 0.33325195F, 1e-8);
  EXPECT_EQ(read_float(f16, DType::F16, 1), 5.9604645e-08F);  // 2^-24
  EXPECT_TRUE(std::isinf(read_float(f16, DType::F16, 2)));
}

TEST(Safetensors, OpenReportsMissingFiles) {
  auto st = SafetensorsFile::open("/nonexistent/model.safetensors");
  ASSERT_FALSE(st);
  EXPECT_NE(st.error().find("cannot open"), std::string::npos);
}
