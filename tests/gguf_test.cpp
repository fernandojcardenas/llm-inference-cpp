#include "llmi/model/gguf.hpp"

#include <gtest/gtest.h>

#include <cstring>

using namespace llmi;
using namespace llmi::gguf;

namespace {

// A tiny hand-rolled GGUF v3 writer, used only to build test inputs -- the
// engine itself never writes GGUF, only reads it (M6's own scope).
class Builder {
 public:
  Builder() {
    push_bytes("GGUF", 4);
    push_u32(3);  // version
  }

  void push_u8(std::uint8_t v) { buf_.push_back(static_cast<std::byte>(v)); }
  void push_u16(std::uint16_t v) {
    for (int k = 0; k < 2; ++k) push_u8(static_cast<std::uint8_t>(v >> (8 * k)));
  }
  void push_u32(std::uint32_t v) {
    for (int k = 0; k < 4; ++k) push_u8(static_cast<std::uint8_t>(v >> (8U * static_cast<unsigned>(k))));
  }
  void push_u64(std::uint64_t v) {
    for (int k = 0; k < 8; ++k) push_u8(static_cast<std::uint8_t>(v >> (8U * static_cast<unsigned>(k))));
  }
  void push_f32(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    push_u32(b);
  }
  void push_bytes(const char* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) push_u8(static_cast<std::uint8_t>(p[i]));
  }
  void push_string(const std::string& s) {
    push_u64(s.size());
    push_bytes(s.data(), s.size());
  }

  // A metadata entry whose value is a single uint32 (handy for
  // general.alignment and for generic malformed-value tests).
  void metadata_u32(const std::string& key, std::uint32_t value_type, std::uint32_t value) {
    push_string(key);
    push_u32(value_type);
    push_u32(value);
  }
  void metadata_string(const std::string& key, const std::string& value) {
    push_string(key);
    push_u32(8);  // ValueType::String
    push_string(value);
  }

  // One F32 tensor's info entry (name, 1-D shape, type=F32, offset); the
  // header section only, not its data bytes.
  void tensor_info_1d(const std::string& name, std::uint64_t n, std::uint64_t offset) {
    push_string(name);
    push_u32(1);  // n_dimensions
    push_u64(n);
    push_u32(0);  // TensorType::F32
    push_u64(offset);
  }

  // Pads to 32-byte alignment (the default) and appends the data section,
  // matching how a real file lays out the gap between the header and the
  // tensor data.
  [[nodiscard]] std::vector<std::byte> finish(std::vector<std::byte> data_section) const {
    std::vector<std::byte> out = buf_;
    while (out.size() % 32 != 0) out.push_back(std::byte{0});
    out.insert(out.end(), data_section.begin(), data_section.end());
    return out;
  }

  [[nodiscard]] const std::vector<std::byte>& raw() const { return buf_; }

 private:
  std::vector<std::byte> buf_;
};

// Builds a minimal valid file: no metadata, one F32 tensor named "w" with
// `n` elements (4*n bytes), default alignment (32).
std::vector<std::byte> make_simple_file(std::uint64_t n = 4) {
  Builder b;
  b.push_u64(1);  // tensor_count
  b.push_u64(0);  // metadata_kv_count
  b.tensor_info_1d("w", n, 0);
  std::vector<std::byte> data(n * 4);
  for (std::size_t i = 0; i < data.size(); ++i) data[i] = std::byte(i & 0xFFU);
  return b.finish(data);
}

Result<GGUFFile> parse(const std::vector<std::byte>& f) { return GGUFFile::parse(f); }

}  // namespace

TEST(Gguf, ParsesAMinimalValidFile) {
  const auto bytes = make_simple_file(4);
  auto f = parse(bytes);
  ASSERT_TRUE(f) << f.error();
  EXPECT_EQ(f->version(), 3U);
  EXPECT_EQ(f->alignment(), 32U);
  ASSERT_EQ(f->tensors().size(), 1U);
  const gguf::TensorInfo* w = f->find("w");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->type, TensorType::F32);
  EXPECT_EQ(w->numel(), 4U);
  EXPECT_EQ(f->data(*w).size(), 16U);
  EXPECT_EQ(f->find("missing"), nullptr);
}

TEST(Gguf, RejectsBadMagicAndVersion) {
  auto f = make_simple_file();
  auto bad_magic = f;
  bad_magic[0] = std::byte{'X'};
  EXPECT_FALSE(parse(bad_magic));

  // version at bytes [4,8)
  auto bad_version = f;
  bad_version[4] = std::byte{2};
  EXPECT_FALSE(parse(bad_version));

  EXPECT_FALSE(parse(std::vector<std::byte>(10)));  // far shorter than the fixed header
}

TEST(Gguf, RejectsCountsBeyondLimits) {
  Builder b;
  b.push_u64(1);                            // tensor_count: fine
  b.push_u64(~0ULL);                        // metadata_kv_count: absurd
  auto bytes = b.raw();
  EXPECT_FALSE(parse(bytes));
}

TEST(Gguf, RejectsTruncationAtEveryStageOfTheHeader) {
  const auto full = make_simple_file();
  // Any prefix shorter than the full file must be rejected, not read past.
  for (std::size_t len = 0; len < full.size(); len += 7) {
    EXPECT_FALSE(parse(std::vector<std::byte>(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(len))))
        << "accepted a truncated file of length " << len;
  }
  EXPECT_TRUE(parse(full));
}

TEST(Gguf, ReadsScalarAndArrayMetadataOfEveryType) {
  Builder b;
  b.push_u64(0);  // tensor_count
  b.push_u64(4);  // metadata_kv_count
  b.metadata_string("general.architecture", "llama");
  b.metadata_u32("llama.block_count", 4, 30);  // UInt32
  // a Float32 scalar
  b.push_string("rope.freq_base");
  b.push_u32(6);  // Float32
  b.push_f32(10000.0F);
  // a UInt32 array [1,2,3]
  b.push_string("tokenizer.ids");
  b.push_u32(9);   // Array
  b.push_u32(4);   // element type UInt32
  b.push_u64(3);   // length
  b.push_u32(1);
  b.push_u32(2);
  b.push_u32(3);
  auto bytes = b.finish({});
  auto f = parse(bytes);
  ASSERT_TRUE(f) << f.error();
  EXPECT_EQ(f->metadata().size(), 4U);
  EXPECT_EQ(f->find_metadata("general.architecture")->as_string, "llama");
  EXPECT_EQ(f->find_metadata("llama.block_count")->as_u64(), 30U);
  EXPECT_FLOAT_EQ(static_cast<float>(f->find_metadata("rope.freq_base")->as_float), 10000.0F);
  const auto& arr = f->find_metadata("tokenizer.ids")->as_array;
  ASSERT_EQ(arr.size(), 3U);
  EXPECT_EQ(arr[0].as_u64(), 1U);
  EXPECT_EQ(arr[2].as_u64(), 3U);
}

TEST(Gguf, RejectsNestedArrays) {
  Builder b;
  b.push_u64(0);
  b.push_u64(1);
  b.push_string("bad");
  b.push_u32(9);  // Array
  b.push_u32(9);  // element type Array -- not allowed
  b.push_u64(0);  // length (never reached)
  EXPECT_FALSE(parse(b.raw()));
}

TEST(Gguf, RejectsUnknownValueAndTensorTypes) {
  Builder b1;
  b1.push_u64(0);
  b1.push_u64(1);
  b1.push_string("k");
  b1.push_u32(999);  // unknown value type
  EXPECT_FALSE(parse(b1.raw()));

  Builder b2;
  b2.push_u64(1);
  b2.push_u64(0);
  b2.push_string("w");
  b2.push_u32(1);
  b2.push_u64(4);
  b2.push_u32(999);  // unknown tensor type
  b2.push_u64(0);
  EXPECT_FALSE(parse(b2.raw()));
}

TEST(Gguf, RejectsDuplicateKeysAndNames) {
  Builder b1;
  b1.push_u64(0);
  b1.push_u64(2);
  b1.metadata_u32("k", 4, 1);
  b1.metadata_u32("k", 4, 2);
  EXPECT_FALSE(parse(b1.finish({})));

  Builder b2;
  b2.push_u64(2);
  b2.push_u64(0);
  b2.tensor_info_1d("w", 4, 0);
  b2.tensor_info_1d("w", 4, 32);
  std::vector<std::byte> data(64);
  EXPECT_FALSE(parse(b2.finish(data)));
}

TEST(Gguf, RejectsBadAlignment) {
  for (std::uint32_t bad : {0U, 3U, 5U, 1U << 30U}) {
    Builder b;
    b.push_u64(0);
    b.push_u64(1);
    b.metadata_u32("general.alignment", 4, bad);
    EXPECT_FALSE(parse(b.finish({}))) << "accepted alignment " << bad;
  }
  // A valid non-default alignment (64) is honored, and tensor offsets are
  // checked against it rather than the default 32.
  Builder ok;
  ok.push_u64(1);
  ok.push_u64(1);
  ok.metadata_u32("general.alignment", 4, 64);
  ok.tensor_info_1d("w", 4, 0);
  auto bytes = ok.raw();
  while (bytes.size() % 64 != 0) bytes.push_back(std::byte{0});
  std::vector<std::byte> data(16);
  bytes.insert(bytes.end(), data.begin(), data.end());
  auto f = parse(bytes);
  ASSERT_TRUE(f) << f.error();
  EXPECT_EQ(f->alignment(), 64U);
}

TEST(Gguf, RejectsMisalignedOffsetsAndOutOfBoundsOffsets) {
  Builder b1;
  b1.push_u64(1);
  b1.push_u64(0);
  b1.tensor_info_1d("w", 4, 4);  // offset 4 is not a multiple of 32
  std::vector<std::byte> data(16);
  EXPECT_FALSE(parse(b1.finish(data)));

  Builder b2;
  b2.push_u64(1);
  b2.push_u64(0);
  b2.tensor_info_1d("w", 4, 1ULL << 40U);  // wildly past the end
  EXPECT_FALSE(parse(b2.finish({})));
}

TEST(Gguf, RejectsOverlappingTensorsButAllowsAlignmentGaps) {
  Builder overlap;
  overlap.push_u64(2);
  overlap.push_u64(0);
  overlap.tensor_info_1d("a", 8, 0);   // bytes [0,32)
  overlap.tensor_info_1d("b", 4, 16);  // bytes [16,32): overlaps a
  std::vector<std::byte> data(32);
  EXPECT_FALSE(parse(overlap.finish(data)));

  // A gap left by alignment padding between tensors is fine (unlike
  // safetensors, which requires exact tiling -- see ADR 0007).
  Builder gap;
  gap.push_u64(2);
  gap.push_u64(0);
  gap.tensor_info_1d("a", 4, 0);    // 16 bytes, needs [0,16)
  gap.tensor_info_1d("b", 4, 32);   // next tensor starts at the next aligned offset, not 16
  std::vector<std::byte> data2(64);
  const auto gap_bytes = gap.finish(data2);
  auto f = parse(gap_bytes);
  EXPECT_TRUE(f) << f.error();
}

TEST(Gguf, RejectsShapesThatOverflowOrAreNotBlockAligned) {
  // rank beyond max_rank (4)
  Builder b1;
  b1.push_u64(1);
  b1.push_u64(0);
  b1.push_string("w");
  b1.push_u32(5);  // n_dimensions = 5 > max_rank
  for (int i = 0; i < 5; ++i) b1.push_u64(1);
  b1.push_u32(0);
  b1.push_u64(0);
  EXPECT_FALSE(parse(b1.raw()));

  // numel overflow: two huge dims
  Builder b2;
  b2.push_u64(1);
  b2.push_u64(0);
  b2.push_string("w");
  b2.push_u32(2);
  b2.push_u64(1ULL << 40U);
  b2.push_u64(1ULL << 40U);
  b2.push_u32(0);  // F32
  b2.push_u64(0);
  EXPECT_FALSE(parse(b2.raw()));

  // a quantized type whose element count isn't a multiple of its block size
  Builder b3;
  b3.push_u64(1);
  b3.push_u64(0);
  b3.push_string("w");
  b3.push_u32(1);
  b3.push_u64(5);  // not a multiple of 32
  b3.push_u32(8);  // Q8_0
  b3.push_u64(0);
  EXPECT_FALSE(parse(b3.raw()));
}

TEST(Gguf, RejectsOversizedAndInvalidUtf8Strings) {
  Limits tight;
  tight.max_string_bytes = 4;
  Builder b;
  b.push_u64(0);
  b.push_u64(1);
  b.metadata_string("k", "too long");
  EXPECT_FALSE(GGUFFile::parse(b.finish({}), tight));

  Builder bad_utf8;
  bad_utf8.push_u64(0);
  bad_utf8.push_u64(1);
  bad_utf8.push_string("k");
  bad_utf8.push_u32(8);  // String
  bad_utf8.push_u64(2);
  bad_utf8.push_u8(0xFF);  // not valid UTF-8
  bad_utf8.push_u8(0xFF);
  EXPECT_FALSE(parse(bad_utf8.raw()));
}

TEST(Gguf, RejectsInvalidBoolValues) {
  Builder b;
  b.push_u64(0);
  b.push_u64(1);
  b.push_string("k");
  b.push_u32(7);  // Bool
  b.push_u8(2);   // neither 0 nor 1
  EXPECT_FALSE(parse(b.raw()));
}

TEST(Gguf, OpenReportsMissingFiles) {
  auto f = GGUFFile::open("/nonexistent/model.gguf");
  ASSERT_FALSE(f);
  EXPECT_NE(f.error().find("cannot open"), std::string::npos);
}

// ---------------------------------------------------------------- conversion

TEST(Gguf, ToF32ConvertsAnF32Tensor) {
  const auto bytes = make_simple_file(4);
  auto f = parse(bytes);
  ASSERT_TRUE(f) << f.error();
  auto vals = to_f32(*f, *f->find("w"));
  ASSERT_TRUE(vals) << vals.error();
  EXPECT_EQ(vals->size(), 4U);
}

TEST(Gguf, ToF32RejectsAQuantizedTensor) {
  Builder b;
  b.push_u64(1);
  b.push_u64(0);
  b.push_string("w");
  b.push_u32(1);
  b.push_u64(32);
  b.push_u32(8);  // Q8_0
  b.push_u64(0);
  std::vector<std::byte> data(34);
  const auto bytes = b.finish(data);
  auto f = parse(bytes);
  ASSERT_TRUE(f) << f.error();
  EXPECT_FALSE(to_f32(*f, *f->find("w")));
}

namespace {
// f16 bit pattern for 1.0: sign 0, exponent 15 (01111), mantissa 0 -> 0x3C00
constexpr std::uint16_t kF16One = 0x3C00;
}  // namespace

TEST(Gguf, ToQuantizedQ8_0CopiesCodesInOrder) {
  // One block of Q8_0: scale 1.0, codes 0..31 in order (ggml and this
  // engine agree on Q8_0's element order, unlike Q4_0 -- see gguf.cpp).
  Builder b;
  b.push_u64(1);
  b.push_u64(0);
  b.push_string("w");
  b.push_u32(1);
  b.push_u64(32);
  b.push_u32(8);  // Q8_0
  b.push_u64(0);
  std::vector<std::byte> data;
  data.reserve(34);
  const auto push16 = [&](std::uint16_t v) {
    data.push_back(std::byte(v & 0xFFU));
    data.push_back(std::byte((v >> 8U) & 0xFFU));
  };
  push16(kF16One);
  for (int i = 0; i < 32; ++i) data.push_back(std::byte(static_cast<std::uint8_t>(i)));
  const auto bytes = b.finish(data);
  auto f = parse(bytes);
  ASSERT_TRUE(f) << f.error();
  auto m = to_quantized(*f, *f->find("w"), /*out=*/1, /*in=*/32);
  ASSERT_TRUE(m) << m.error();
  EXPECT_FLOAT_EQ(m->scales[0], 1.0F);
  std::vector<float> row(32);
  quant::dequantize_row(*m, 0, row.data());
  for (int i = 0; i < 32; ++i) EXPECT_FLOAT_EQ(row[static_cast<std::size_t>(i)], static_cast<float>(i));
}

TEST(Gguf, ToQuantizedQ4_0UnpacksGgufsHalfBlockPairingNotAdjacentPairs) {
  // ggml packs byte j as (low nibble = element j, high nibble = element
  // j+16). Byte 0 = 0x91 means element 0 has nibble 1, element 16 has
  // nibble 9; with scale 2.0 that's value (1-8)*2=-14 at index 0 and
  // (9-8)*2=2 at index 16. A naive adjacent-pair memcpy would instead read
  // this as elements 0 and 1 -- this test fails under that bug.
  Builder b;
  b.push_u64(1);
  b.push_u64(0);
  b.push_string("w");
  b.push_u32(1);
  b.push_u64(32);
  b.push_u32(2);  // Q4_0
  b.push_u64(0);
  std::vector<std::byte> data;
  const auto push16 = [&](std::uint16_t v) {
    data.push_back(std::byte(v & 0xFFU));
    data.push_back(std::byte((v >> 8U) & 0xFFU));
  };
  // f16 for 2.0: sign 0, exponent 16 (10000), mantissa 0 -> 0x4000
  push16(0x4000);
  data.push_back(std::byte{0x91});  // byte 0: lo=1 (elem 0), hi=9 (elem 16)
  for (int i = 1; i < 16; ++i) data.push_back(std::byte{0x88});  // lo=hi=8 -> value 0 everywhere else
  const auto bytes = b.finish(data);
  auto f = parse(bytes);
  ASSERT_TRUE(f) << f.error();
  auto m = to_quantized(*f, *f->find("w"), 1, 32);
  ASSERT_TRUE(m) << m.error();
  std::vector<float> row(32);
  quant::dequantize_row(*m, 0, row.data());
  EXPECT_FLOAT_EQ(row[0], -14.0F);
  EXPECT_FLOAT_EQ(row[16], 2.0F);
  EXPECT_FLOAT_EQ(row[1], 0.0F);
  EXPECT_FLOAT_EQ(row[17], 0.0F);
}

TEST(Gguf, ToQuantizedRejectsShapeMismatches) {
  const auto bytes = make_simple_file(4);
  auto f = parse(bytes);
  ASSERT_TRUE(f) << f.error();
  EXPECT_FALSE(to_quantized(*f, *f->find("w"), 2, 2));  // "w" is F32, not Q8_0/Q4_0
}
