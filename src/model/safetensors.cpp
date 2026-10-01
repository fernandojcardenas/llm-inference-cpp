#include "llmi/model/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <optional>
#include <system_error>

#include "llmi/util/json.hpp"

namespace llmi {

// ---------------------------------------------------------------- MappedFile

namespace {
// Thread-safe, unlike std::strerror.
std::string errno_message(int err) { return std::error_code(err, std::generic_category()).message(); }
}  // namespace

Result<std::shared_ptr<MappedFile>> MappedFile::open(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return fail("cannot open " + path + ": " + errno_message(errno));
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    const int err = errno;
    ::close(fd);
    return fail("cannot stat " + path + ": " + errno_message(err));
  }
  if (!S_ISREG(st.st_mode)) {
    ::close(fd);
    return fail(path + " is not a regular file");
  }
  const auto size = static_cast<std::size_t>(st.st_size);
  const std::byte* data = nullptr;
  if (size > 0) {
    void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
      const int err = errno;
      ::close(fd);
      return fail("cannot map " + path + ": " + errno_message(err));
    }
    data = static_cast<const std::byte*>(p);
  }
  ::close(fd);  // the mapping stays valid after the descriptor is closed
  return std::shared_ptr<MappedFile>(new MappedFile(data, size));
}

MappedFile::~MappedFile() {
  if (data_ != nullptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): munmap takes void*
    ::munmap(const_cast<std::byte*>(data_), size_);
  }
}

// ---------------------------------------------------------------- dtypes

namespace {

struct DTypeEntry {
  std::string_view name;
  DType type;
  std::size_t size;
};

constexpr DTypeEntry kDTypes[] = {
    {"F64", DType::F64, 8},  {"F32", DType::F32, 4},   {"F16", DType::F16, 2},
    {"BF16", DType::BF16, 2}, {"I64", DType::I64, 8},   {"I32", DType::I32, 4},
    {"I16", DType::I16, 2},  {"I8", DType::I8, 1},     {"U8", DType::U8, 1},
    {"BOOL", DType::Bool, 1}, {"F8_E4M3", DType::F8_E4M3, 1}, {"F8_E5M2", DType::F8_E5M2, 1},
};

std::optional<DType> parse_dtype(std::string_view s) {
  for (const auto& e : kDTypes) {
    if (e.name == s) return e.type;
  }
  return std::nullopt;
}

// a * b, or nullopt on overflow.
std::optional<std::uint64_t> mul(std::uint64_t a, std::uint64_t b) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return std::nullopt;
  return a * b;
}

}  // namespace

std::string_view dtype_name(DType t) {
  for (const auto& e : kDTypes) {
    if (e.type == t) return e.name;
  }
  return "?";
}

std::size_t dtype_size(DType t) {
  for (const auto& e : kDTypes) {
    if (e.type == t) return e.size;
  }
  return 0;
}

std::uint64_t TensorInfo::numel() const {
  std::uint64_t n = 1;
  for (const auto d : shape) n *= d;  // checked against the byte range at parse time
  return n;
}

// ---------------------------------------------------------------- parsing

Result<SafetensorsFile> SafetensorsFile::parse(std::span<const std::byte> bytes,
                                               const SafetensorsLimits& limits) {
  if (bytes.size() < 8) return fail("safetensors: file shorter than the 8-byte header size");
  std::uint64_t header_len = 0;
  for (int k = 7; k >= 0; --k) {
    header_len = (header_len << 8U) | std::to_integer<std::uint64_t>(bytes[static_cast<std::size_t>(k)]);
  }
  if (header_len > limits.max_header_bytes) {
    return fail("safetensors: header size " + std::to_string(header_len) + " exceeds the limit");
  }
  if (header_len > bytes.size() - 8) return fail("safetensors: header extends past the end of the file");

  const auto header_bytes = bytes.subspan(8, static_cast<std::size_t>(header_len));
  const std::string_view header(reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());
  json::Limits jl;
  jl.max_depth = 8;
  auto root = json::parse(header, jl);
  if (!root) return fail("safetensors: header: " + root.error());
  if (!root->is_object()) return fail("safetensors: header is not a JSON object");

  SafetensorsFile f;
  f.data_section_ = bytes.subspan(8 + static_cast<std::size_t>(header_len));
  const std::uint64_t data_size = f.data_section_.size();

  for (const auto& [name, v] : root->members()) {
    if (name == "__metadata__") {
      if (!v.is_object()) return fail("safetensors: __metadata__ is not an object");
      for (const auto& [mk, mv] : v.members()) {
        if (!mv.is_string()) return fail("safetensors: __metadata__ value for \"" + mk + "\" is not a string");
        f.metadata_.emplace(mk, mv.text());
      }
      continue;
    }
    if (f.tensors_.size() >= limits.max_tensors) return fail("safetensors: too many tensors");
    if (name.empty() || name.size() > limits.max_name_bytes) return fail("safetensors: bad tensor name length");
    const std::string where = "safetensors: tensor \"" + name + "\": ";
    if (!v.is_object()) return fail(where + "entry is not an object");

    TensorInfo t;
    t.name = name;
    bool have_dtype = false;
    bool have_shape = false;
    bool have_offsets = false;
    for (const auto& [field, fv] : v.members()) {
      if (field == "dtype") {
        auto dt = fv.is_string() ? parse_dtype(fv.text()) : std::nullopt;
        if (!dt) return fail(where + "unknown dtype");
        t.dtype = *dt;
        have_dtype = true;
      } else if (field == "shape") {
        if (!fv.is_array()) return fail(where + "shape is not an array");
        if (fv.items().size() > limits.max_rank) return fail(where + "too many dimensions");
        for (const auto& d : fv.items()) {
          auto n = d.as_u64();
          if (!n) return fail(where + "shape entry is not a non-negative integer");
          t.shape.push_back(*n);
        }
        have_shape = true;
      } else if (field == "data_offsets") {
        if (!fv.is_array() || fv.items().size() != 2) return fail(where + "data_offsets is not [begin, end]");
        auto b = fv.items()[0].as_u64();
        auto e = fv.items()[1].as_u64();
        if (!b || !e) return fail(where + "data_offsets are not non-negative integers");
        t.begin = *b;
        t.end = *e;
        have_offsets = true;
      } else {
        std::string msg = where;
        msg += "unknown field \"";
        msg += field;
        msg += '"';
        return fail(msg);
      }
    }
    if (!have_dtype || !have_shape || !have_offsets) return fail(where + "missing dtype, shape or data_offsets");
    if (t.begin > t.end || t.end > data_size) return fail(where + "data_offsets out of bounds");

    std::optional<std::uint64_t> expected = dtype_size(t.dtype);
    for (const auto d : t.shape) {
      expected = mul(*expected, d);
      if (!expected) return fail(where + "shape overflows");
    }
    if (*expected != t.end - t.begin) {
      return fail(where + "byte range (" + std::to_string(t.end - t.begin) + ") does not match dtype x shape (" +
                  std::to_string(*expected) + ")");
    }
    f.tensors_.push_back(std::move(t));
  }

  // The data section must be covered exactly: no overlaps, no holes, nothing after.
  std::vector<const TensorInfo*> by_offset;
  by_offset.reserve(f.tensors_.size());
  for (const auto& t : f.tensors_) by_offset.push_back(&t);
  std::sort(by_offset.begin(), by_offset.end(), [](const TensorInfo* a, const TensorInfo* b) {
    return a->begin != b->begin ? a->begin < b->begin : a->end < b->end;
  });
  std::uint64_t cursor = 0;
  for (const auto* t : by_offset) {
    if (t->begin < cursor) return fail("safetensors: tensor \"" + t->name + "\" overlaps another tensor");
    if (t->begin > cursor) return fail("safetensors: hole in the data section before \"" + t->name + "\"");
    cursor = t->end;
  }
  if (cursor != data_size) return fail("safetensors: data section has bytes not owned by any tensor");

  std::sort(f.tensors_.begin(), f.tensors_.end(),
            [](const TensorInfo& a, const TensorInfo& b) { return a.name < b.name; });
  return f;
}

Result<SafetensorsFile> SafetensorsFile::open(const std::string& path, const SafetensorsLimits& limits) {
  auto file = MappedFile::open(path);
  if (!file) return fail(file.error());
  auto parsed = parse(file.value()->bytes(), limits);
  if (!parsed) return fail(path + ": " + parsed.error());
  parsed->file_ = std::move(file.value());
  return parsed;
}

const TensorInfo* SafetensorsFile::find(std::string_view name) const {
  auto it = std::lower_bound(tensors_.begin(), tensors_.end(), name,
                             [](const TensorInfo& t, std::string_view n) { return t.name < n; });
  if (it == tensors_.end() || it->name != name) return nullptr;
  return &*it;
}

std::span<const std::byte> SafetensorsFile::data(const TensorInfo& t) const {
  return data_section_.subspan(static_cast<std::size_t>(t.begin), static_cast<std::size_t>(t.end - t.begin));
}

// ---------------------------------------------------------------- conversion

namespace {

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
  std::uint32_t exp = (h >> 10U) & 0x1FU;
  std::uint32_t mant = h & 0x3FFU;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {  // subnormal: normalise
      exp = 127 - 15 + 1;
      while ((mant & 0x400U) == 0) {
        mant <<= 1U;
        --exp;
      }
      mant &= 0x3FFU;
      bits = sign | (exp << 23U) | (mant << 13U);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000U | (mant << 13U);
  } else {
    bits = sign | ((exp + 127 - 15) << 23U) | (mant << 13U);
  }
  return std::bit_cast<float>(bits);
}

}  // namespace

float read_float(std::span<const std::byte> raw, DType dtype, std::size_t i) {
  switch (dtype) {
    case DType::F32: {
      std::uint32_t b = 0;
      std::memcpy(&b, raw.data() + i * 4, 4);
      if constexpr (std::endian::native == std::endian::big) b = __builtin_bswap32(b);
      return std::bit_cast<float>(b);
    }
    case DType::BF16: {
      std::uint16_t b = 0;
      std::memcpy(&b, raw.data() + i * 2, 2);
      if constexpr (std::endian::native == std::endian::big) b = __builtin_bswap16(b);
      return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16U);
    }
    case DType::F16: {
      std::uint16_t b = 0;
      std::memcpy(&b, raw.data() + i * 2, 2);
      if constexpr (std::endian::native == std::endian::big) b = __builtin_bswap16(b);
      return half_to_float(b);
    }
    default:
      return 0.0F;
  }
}

}  // namespace llmi
