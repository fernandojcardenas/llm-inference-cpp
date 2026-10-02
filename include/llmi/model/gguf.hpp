#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "llmi/model/quant.hpp"
#include "llmi/model/safetensors.hpp"  // MappedFile
#include "llmi/util/result.hpp"

namespace llmi::gguf {

// The tensor element types a GGUF file may declare (ggml_type). This is the
// full set of codes ggml has ever assigned (gaps are codes it removed or has
// not assigned); an unassigned or unrecognized code is rejected rather than
// guessed at. Only F32, F16, Q8_0 and Q4_0 can be turned into data this
// engine can compute with (to_f32, to_quantized) -- the rest are recognized
// well enough to validate their size and offset, matching M1's own stance of
// erroring on what isn't supported rather than silently mishandling it.
enum class TensorType : std::uint8_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
  BF16 = 30,
  TQ1_0 = 34,
  TQ2_0 = 35,
  MXFP4 = 39,
  NVFP4 = 40,
  Q1_0 = 41,
};

std::string_view tensor_type_name(TensorType t);

struct TensorTypeInfo {
  std::size_t block_size;  // elements per block (1 for the unquantized types)
  std::size_t type_size;   // bytes per block
};
// nullopt for any code not in the table above.
std::optional<TensorTypeInfo> tensor_type_info(TensorType t);

// GGUF's own metadata value-kind tags (what the file format calls
// `gguf_metadata_value_type`).
enum class ValueType : std::uint8_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

// One metadata value. Scalars are widened into as_uint/as_int/as_float so
// callers don't need a type-specific accessor per case; `type` says which of
// as_uint/as_int/as_float/as_bool/as_string/as_array is meaningful. Arrays
// are one level deep only: GGUF does not allow an array of arrays, and this
// parser rejects one rather than silently flattening or truncating it.
struct Value {
  ValueType type = ValueType::UInt8;
  std::uint64_t as_uint = 0;
  std::int64_t as_int = 0;
  double as_float = 0;
  bool as_bool = false;
  std::string as_string;
  ValueType array_type = ValueType::UInt8;  // meaningful when type == Array
  std::vector<Value> as_array;

  // Any integer-kind scalar (UInt*/Int*), if it is non-negative.
  [[nodiscard]] std::optional<std::uint64_t> as_u64() const;
};

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> shape;  // ne[0..n_dims): GGUF's own order, ne[0] fastest-varying
  TensorType type = TensorType::F32;
  std::uint64_t offset = 0;  // from the start of the (alignment-padded) data section
  std::uint64_t size = 0;    // byte size; computed and overflow-checked at parse time

  [[nodiscard]] std::uint64_t numel() const;
};

// Limits applied while reading a file, checked before anything they bound is
// allocated or indexed. GGUF files describe their own sizes in-band (tensor
// and metadata counts, string and array lengths); a malicious or merely
// corrupt file can claim any of them, so every one is checked against both a
// limit and the bytes actually remaining before it is trusted.
struct Limits {
  std::uint64_t max_header_bytes = 512ULL * 1024 * 1024;  // metadata + tensor-info section
  std::size_t max_tensors = 1'000'000;
  std::size_t max_metadata_entries = 1'000'000;
  std::size_t max_rank = 4;                  // GGML_MAX_DIMS in every ggml release to date
  std::size_t max_name_bytes = 64;           // GGML_MAX_NAME in every ggml release to date
  std::size_t max_string_bytes = 1 << 20;     // 1 MiB; real metadata keys/strings are tiny
  std::size_t max_array_len = 10'000'000;
  std::uint64_t max_alignment = 1ULL << 20;  // sanity bound on general.alignment
};

// A parsed GGUF v3 file (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
// Layout: magic "GGUF", version (must be 3; v1/v2 used 32-bit lengths and are
// not read by this parser), tensor_count, metadata_kv_count, that many
// metadata entries, tensor_count tensor-info entries, padding to `alignment`
// (default 32, overridable by a `general.alignment` metadata entry), then
// the tensor data section. Every count, length, offset and size the file
// claims is checked before use: against this parser's own limits, against
// the bytes actually remaining (no read past the end of the input), and
// (for tensor byte sizes) with overflow-checked arithmetic -- the same class
// of bug CVE-2026-27940 and CVE-2026-33298 exploited in llama.cpp's own GGUF
// reader (see docs/gguf-threat-model.md and ADR 0001).
class GGUFFile {
 public:
  static Result<GGUFFile> parse(std::span<const std::byte> bytes, const Limits& limits = {});
  static Result<GGUFFile> open(const std::string& path, const Limits& limits = {});

  [[nodiscard]] std::uint32_t version() const { return version_; }
  [[nodiscard]] std::uint64_t alignment() const { return alignment_; }
  [[nodiscard]] const std::map<std::string, Value>& metadata() const { return metadata_; }
  [[nodiscard]] const Value* find_metadata(std::string_view key) const;
  [[nodiscard]] const std::vector<TensorInfo>& tensors() const { return tensors_; }
  [[nodiscard]] const TensorInfo* find(std::string_view name) const;
  // The tensor's raw bytes, exactly as the file stores them (GGUF's own
  // per-type block layout -- see to_f32/to_quantized to get engine-usable
  // floats or a quant::QuantizedMatrix out of them).
  [[nodiscard]] std::span<const std::byte> data(const TensorInfo& t) const;

 private:
  std::shared_ptr<MappedFile> file_;
  std::uint32_t version_ = 0;
  std::uint64_t alignment_ = 32;
  std::map<std::string, Value> metadata_;
  std::vector<TensorInfo> tensors_;  // sorted by name
  std::span<const std::byte> data_section_;
};

// Converts an F32, F16 or BF16 tensor's raw bytes into a plain float vector,
// in the order GGUF stores them (ne[0] fastest).
Result<std::vector<float>> to_f32(const GGUFFile& f, const TensorInfo& t);

// Converts a Q8_0 or Q4_0 tensor into this engine's own planar
// quant::QuantizedMatrix (src/model/quant.cpp). GGUF's on-disk block layout
// (a float16 scale followed by the block's packed values, repeated per
// block) is byte-for-byte what llama.cpp itself writes; it differs from
// this engine's in-memory layout (a separate scales array, documented in
// quant.hpp), which is what this function converts between. `out`/`in` are
// the engine's row-major (out, in) convention; GGUF stores a 2-D weight as
// ne = [in, out] (ne[0] fastest), the reverse order -- see
// docs/gguf.md for why.
Result<quant::QuantizedMatrix> to_quantized(const GGUFFile& f, const TensorInfo& t, std::size_t out, std::size_t in);

}  // namespace llmi::gguf
