#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "llmi/util/result.hpp"

namespace llmi {

// A read-only memory map of a whole file. Weights are never copied: tensors
// are views into the mapping, and the OS pages them in as they are used.
class MappedFile {
 public:
  static Result<std::shared_ptr<MappedFile>> open(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&&) = delete;
  MappedFile& operator=(MappedFile&&) = delete;

  [[nodiscard]] std::span<const std::byte> bytes() const { return {data_, size_}; }

 private:
  MappedFile(const std::byte* data, std::size_t size) : data_(data), size_(size) {}
  const std::byte* data_;
  std::size_t size_;
};

enum class DType : std::uint8_t { F64, F32, F16, BF16, I64, I32, I16, I8, U8, Bool, F8_E4M3, F8_E5M2 };

std::string_view dtype_name(DType t);
std::size_t dtype_size(DType t);

struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;
  std::vector<std::uint64_t> shape;
  std::uint64_t begin = 0;  // byte offsets into the data section
  std::uint64_t end = 0;

  [[nodiscard]] std::uint64_t numel() const;
};

// Limits applied while reading a file; defaults follow the safetensors
// format's own (100 MB header) and are generous for real models.
struct SafetensorsLimits {
  std::uint64_t max_header_bytes = 100ULL * 1024 * 1024;
  std::size_t max_tensors = 1'000'000;
  std::size_t max_rank = 8;
  std::size_t max_name_bytes = 1024;
};

// A parsed safetensors file (https://github.com/huggingface/safetensors).
// Layout: 8-byte little-endian header size N, N bytes of JSON, then the data
// section. Parsing rejects anything the format does not allow: offsets out
// of bounds or overlapping, holes in the data section, byte ranges that do
// not match dtype x shape (checked without integer overflow), duplicate or
// unknown fields, and unknown dtypes.
class SafetensorsFile {
 public:
  // Parses a file that is already in memory. `bytes` must outlive the result.
  static Result<SafetensorsFile> parse(std::span<const std::byte> bytes, const SafetensorsLimits& limits = {});
  // Maps a file from disk and parses it; the mapping is kept alive.
  static Result<SafetensorsFile> open(const std::string& path, const SafetensorsLimits& limits = {});

  [[nodiscard]] const std::vector<TensorInfo>& tensors() const { return tensors_; }
  [[nodiscard]] const std::map<std::string, std::string>& metadata() const { return metadata_; }
  [[nodiscard]] const TensorInfo* find(std::string_view name) const;
  // The tensor's raw bytes (little-endian, row-major).
  [[nodiscard]] std::span<const std::byte> data(const TensorInfo& t) const;

 private:
  std::shared_ptr<MappedFile> file_;
  std::span<const std::byte> data_section_;
  std::vector<TensorInfo> tensors_;  // sorted by name
  std::map<std::string, std::string> metadata_;
};

// Reads element i of a floating-point tensor as float (F32, F16, BF16).
float read_float(std::span<const std::byte> raw, DType dtype, std::size_t i);

}  // namespace llmi
