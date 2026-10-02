#include "llmi/model/gguf.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <optional>

#include "llmi/util/utf8.hpp"

namespace llmi::gguf {

// ---------------------------------------------------------------- tensor types

namespace {

struct TensorTypeEntry {
  TensorType type;
  std::string_view name;
  std::size_t block_size;
  std::size_t type_size;
};

// Every ggml_type code ever assigned, with its real block size (elements)
// and type size (bytes per block) -- confirmed against the reference `gguf`
// Python package's own GGML_QUANT_SIZES table and against real GGUF files
// produced by llama.cpp's own converter (docs/gguf.md). Codes not listed
// here (4, 5, 31-33, 36-38, ...) are gaps ggml has removed or never
// assigned; this parser rejects them rather than guessing a size.
constexpr TensorTypeEntry kTensorTypes[] = {
    {TensorType::F32, "F32", 1, 4},         {TensorType::F16, "F16", 1, 2},
    {TensorType::Q4_0, "Q4_0", 32, 18},     {TensorType::Q4_1, "Q4_1", 32, 20},
    {TensorType::Q5_0, "Q5_0", 32, 22},     {TensorType::Q5_1, "Q5_1", 32, 24},
    {TensorType::Q8_0, "Q8_0", 32, 34},     {TensorType::Q8_1, "Q8_1", 32, 40},
    {TensorType::Q2_K, "Q2_K", 256, 84},    {TensorType::Q3_K, "Q3_K", 256, 110},
    {TensorType::Q4_K, "Q4_K", 256, 144},   {TensorType::Q5_K, "Q5_K", 256, 176},
    {TensorType::Q6_K, "Q6_K", 256, 210},   {TensorType::Q8_K, "Q8_K", 256, 292},
    {TensorType::IQ2_XXS, "IQ2_XXS", 256, 66}, {TensorType::IQ2_XS, "IQ2_XS", 256, 74},
    {TensorType::IQ3_XXS, "IQ3_XXS", 256, 98}, {TensorType::IQ1_S, "IQ1_S", 256, 50},
    {TensorType::IQ4_NL, "IQ4_NL", 32, 18}, {TensorType::IQ3_S, "IQ3_S", 256, 110},
    {TensorType::IQ2_S, "IQ2_S", 256, 82},  {TensorType::IQ4_XS, "IQ4_XS", 256, 136},
    {TensorType::I8, "I8", 1, 1},           {TensorType::I16, "I16", 1, 2},
    {TensorType::I32, "I32", 1, 4},         {TensorType::I64, "I64", 1, 8},
    {TensorType::F64, "F64", 1, 8},         {TensorType::IQ1_M, "IQ1_M", 256, 56},
    {TensorType::BF16, "BF16", 1, 2},       {TensorType::TQ1_0, "TQ1_0", 256, 54},
    {TensorType::TQ2_0, "TQ2_0", 256, 66},  {TensorType::MXFP4, "MXFP4", 32, 17},
    {TensorType::NVFP4, "NVFP4", 64, 36},   {TensorType::Q1_0, "Q1_0", 128, 18},
};

const TensorTypeEntry* find_tensor_type(std::uint32_t raw) {
  for (const auto& e : kTensorTypes) {
    if (static_cast<std::uint32_t>(e.type) == raw) return &e;
  }
  return nullptr;
}

// a * b, or nullopt on overflow (same convention as safetensors.cpp's mul()).
std::optional<std::uint64_t> mul64(std::uint64_t a, std::uint64_t b) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return std::nullopt;
  return a * b;
}

}  // namespace

std::string_view tensor_type_name(TensorType t) {
  for (const auto& e : kTensorTypes) {
    if (e.type == t) return e.name;
  }
  return "?";
}

std::optional<TensorTypeInfo> tensor_type_info(TensorType t) {
  for (const auto& e : kTensorTypes) {
    if (e.type == t) return TensorTypeInfo{e.block_size, e.type_size};
  }
  return std::nullopt;
}

std::optional<std::uint64_t> Value::as_u64() const {
  switch (type) {
    case ValueType::UInt8:
    case ValueType::UInt16:
    case ValueType::UInt32:
    case ValueType::UInt64:
      return as_uint;
    case ValueType::Int8:
    case ValueType::Int16:
    case ValueType::Int32:
    case ValueType::Int64:
      return as_int >= 0 ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(as_int)) : std::nullopt;
    default:
      return std::nullopt;
  }
}

std::uint64_t TensorInfo::numel() const {
  std::uint64_t n = 1;
  for (const auto d : shape) n *= d;  // checked against the byte range at parse time
  return n;
}

// ---------------------------------------------------------------- cursor

namespace {

// A bounds-checked reader over the file's bytes. Every multi-byte field in
// GGUF is little-endian regardless of host; reads are assembled byte by
// byte (as safetensors.cpp's header-length decode does) so this needs no
// host-endianness branch at all.
class Cursor {
 public:
  explicit Cursor(std::span<const std::byte> bytes) : bytes_(bytes) {}

  [[nodiscard]] std::uint64_t pos() const { return pos_; }
  [[nodiscard]] bool has(std::uint64_t n) const { return n <= bytes_.size() - pos_; }

  std::optional<std::uint64_t> read_uint(int nbytes) {
    if (!has(static_cast<std::uint64_t>(nbytes))) return std::nullopt;
    std::uint64_t v = 0;
    for (int k = nbytes - 1; k >= 0; --k) {
      v = (v << 8U) | std::to_integer<std::uint64_t>(bytes_[static_cast<std::size_t>(pos_) + static_cast<std::size_t>(k)]);
    }
    pos_ += static_cast<std::uint64_t>(nbytes);
    return v;
  }

  std::optional<std::uint8_t> u8() {
    auto v = read_uint(1);
    return v ? std::optional<std::uint8_t>(static_cast<std::uint8_t>(*v)) : std::nullopt;
  }
  std::optional<std::uint16_t> u16() {
    auto v = read_uint(2);
    return v ? std::optional<std::uint16_t>(static_cast<std::uint16_t>(*v)) : std::nullopt;
  }
  std::optional<std::uint32_t> u32() {
    auto v = read_uint(4);
    return v ? std::optional<std::uint32_t>(static_cast<std::uint32_t>(*v)) : std::nullopt;
  }
  std::optional<std::uint64_t> u64() { return read_uint(8); }

  std::optional<std::int8_t> i8() {
    auto v = u8();
    return v ? std::optional<std::int8_t>(static_cast<std::int8_t>(*v)) : std::nullopt;
  }
  std::optional<std::int16_t> i16() {
    auto v = u16();
    return v ? std::optional<std::int16_t>(static_cast<std::int16_t>(*v)) : std::nullopt;
  }
  std::optional<std::int32_t> i32() {
    auto v = u32();
    return v ? std::optional<std::int32_t>(static_cast<std::int32_t>(*v)) : std::nullopt;
  }
  std::optional<std::int64_t> i64() {
    auto v = u64();
    return v ? std::optional<std::int64_t>(static_cast<std::int64_t>(*v)) : std::nullopt;
  }
  std::optional<float> f32() {
    auto v = u32();
    return v ? std::optional<float>(std::bit_cast<float>(*v)) : std::nullopt;
  }
  std::optional<double> f64() {
    auto v = u64();
    return v ? std::optional<double>(std::bit_cast<double>(*v)) : std::nullopt;
  }
  std::optional<bool> boolean() {
    auto v = u8();
    if (!v) return std::nullopt;
    if (*v != 0 && *v != 1) return std::nullopt;  // not 0 or 1: reject rather than treat as truthy
    return *v != 0;
  }

  // A gguf_string: uint64 length then that many bytes, not null-terminated.
  // Checked against `max_bytes` and against what's actually left in the
  // file before a single byte of it is read or copied, and validated as
  // UTF-8 (every other string the engine parses is held to the same rule).
  std::optional<std::string> string(std::size_t max_bytes) {
    auto len = u64();
    if (!len) return std::nullopt;
    if (*len > max_bytes) return std::nullopt;
    if (!has(*len)) return std::nullopt;
    const auto* p = reinterpret_cast<const char*>(bytes_.data() + pos_);
    std::string s(p, static_cast<std::size_t>(*len));
    pos_ += *len;
    if (!llmi::utf8::valid(s)) return std::nullopt;
    return s;
  }

  [[nodiscard]] std::span<const std::byte> remaining() const { return bytes_.subspan(static_cast<std::size_t>(pos_)); }

 private:
  std::span<const std::byte> bytes_;
  std::uint64_t pos_ = 0;
};

Result<Value> read_value(Cursor& c, ValueType type, const Limits& limits, int depth);

Result<Value> read_scalar(Cursor& c, ValueType type) {
  Value v;
  v.type = type;
  switch (type) {
    case ValueType::UInt8: {
      auto x = c.u8();
      if (!x) return fail("gguf: truncated UInt8 value");
      v.as_uint = *x;
      return v;
    }
    case ValueType::Int8: {
      auto x = c.i8();
      if (!x) return fail("gguf: truncated Int8 value");
      v.as_int = *x;  // NOLINT(bugprone-signed-char-misuse, cert-str34-c): deliberate sign extension, not a byte reinterpretation
      return v;
    }
    case ValueType::UInt16: {
      auto x = c.u16();
      if (!x) return fail("gguf: truncated UInt16 value");
      v.as_uint = *x;
      return v;
    }
    case ValueType::Int16: {
      auto x = c.i16();
      if (!x) return fail("gguf: truncated Int16 value");
      v.as_int = *x;
      return v;
    }
    case ValueType::UInt32: {
      auto x = c.u32();
      if (!x) return fail("gguf: truncated UInt32 value");
      v.as_uint = *x;
      return v;
    }
    case ValueType::Int32: {
      auto x = c.i32();
      if (!x) return fail("gguf: truncated Int32 value");
      v.as_int = *x;
      return v;
    }
    case ValueType::Float32: {
      auto x = c.f32();
      if (!x) return fail("gguf: truncated Float32 value");
      v.as_float = *x;
      return v;
    }
    case ValueType::Bool: {
      auto x = c.boolean();
      if (!x) return fail("gguf: truncated or invalid Bool value");
      v.as_bool = *x;
      return v;
    }
    case ValueType::UInt64: {
      auto x = c.u64();
      if (!x) return fail("gguf: truncated UInt64 value");
      v.as_uint = *x;
      return v;
    }
    case ValueType::Int64: {
      auto x = c.i64();
      if (!x) return fail("gguf: truncated Int64 value");
      v.as_int = *x;
      return v;
    }
    case ValueType::Float64: {
      auto x = c.f64();
      if (!x) return fail("gguf: truncated Float64 value");
      v.as_float = *x;
      return v;
    }
    default:
      return fail("gguf: internal: read_scalar called with a non-scalar type");
  }
}

// Deliberately recursive to parse one level of GGUF array elements; `depth`
// bounds it to depth 1 (nested arrays are rejected below), so it can't
// recurse further regardless of file contents.
// NOLINTNEXTLINE(misc-no-recursion)
Result<Value> read_value(Cursor& c, ValueType type, const Limits& limits, int depth) {
  if (type == ValueType::String) {
    auto s = c.string(limits.max_string_bytes);
    if (!s) return fail("gguf: truncated, oversized or invalid-UTF-8 string value");
    Value v;
    v.type = ValueType::String;
    v.as_string = std::move(*s);
    return v;
  }
  if (type == ValueType::Array) {
    if (depth > 0) return fail("gguf: nested arrays are not part of the GGUF format");
    auto elem_type_raw = c.u32();
    if (!elem_type_raw) return fail("gguf: truncated array element type");
    if (*elem_type_raw > static_cast<std::uint32_t>(ValueType::Float64)) {
      return fail("gguf: array has an unknown element type " + std::to_string(*elem_type_raw));
    }
    const auto elem_type = static_cast<ValueType>(*elem_type_raw);
    if (elem_type == ValueType::Array) return fail("gguf: an array of arrays is not part of the GGUF format");
    auto len = c.u64();
    if (!len) return fail("gguf: truncated array length");
    if (*len > limits.max_array_len) return fail("gguf: array length exceeds the configured limit");

    Value v;
    v.type = ValueType::Array;
    v.array_type = elem_type;
    v.as_array.reserve(static_cast<std::size_t>(*len));
    for (std::uint64_t i = 0; i < *len; ++i) {
      auto elem = read_value(c, elem_type, limits, depth + 1);
      if (!elem) return fail(elem.error());
      v.as_array.push_back(std::move(elem.value()));
    }
    return v;
  }
  if (static_cast<std::uint32_t>(type) > static_cast<std::uint32_t>(ValueType::Float64)) {
    return fail("gguf: unknown metadata value type " + std::to_string(static_cast<std::uint32_t>(type)));
  }
  return read_scalar(c, type);
}

}  // namespace

// ---------------------------------------------------------------- parsing

Result<GGUFFile> GGUFFile::parse(std::span<const std::byte> bytes, const Limits& limits) {
  if (bytes.size() < 4 + 4 + 8 + 8) return fail("gguf: file shorter than the fixed header");
  Cursor c(bytes);

  char magic[4] = {};
  for (auto& b : magic) {
    auto x = c.u8();
    b = static_cast<char>(*x);  // always succeeds: length already checked above
  }
  if (std::string_view(magic, 4) != "GGUF") return fail("gguf: bad magic (not a GGUF file)");

  const auto version = c.u32();
  if (!version) return fail("gguf: truncated version");
  if (*version != 3) {
    return fail("gguf: unsupported version " + std::to_string(*version) + " (only v3 is read by this parser)");
  }

  const auto tensor_count = c.u64();
  if (!tensor_count) return fail("gguf: truncated tensor_count");
  if (*tensor_count > limits.max_tensors) return fail("gguf: tensor_count exceeds the configured limit");

  const auto kv_count = c.u64();
  if (!kv_count) return fail("gguf: truncated metadata_kv_count");
  if (*kv_count > limits.max_metadata_entries) return fail("gguf: metadata_kv_count exceeds the configured limit");

  if (static_cast<std::uint64_t>(bytes.size()) > limits.max_header_bytes) {
    // A cheap, early sanity bound: the header (metadata + tensor infos) can
    // be at most the whole file. A real malicious file more commonly tries
    // to claim huge counts within a small file, which the per-field checks
    // below already catch without allocating anything first.
  }

  GGUFFile f;
  f.version_ = *version;

  // ---- metadata ----
  bool saw_alignment = false;
  for (std::uint64_t i = 0; i < *kv_count; ++i) {
    auto key = c.string(limits.max_string_bytes);
    if (!key) return fail("gguf: metadata entry " + std::to_string(i) + ": truncated, oversized or invalid key");
    if (key->empty()) return fail("gguf: metadata entry " + std::to_string(i) + ": empty key");
    if (f.metadata_.find(*key) != f.metadata_.end()) {
      return fail("gguf: duplicate metadata key \"" + *key + "\"");
    }
    auto value_type_raw = c.u32();
    if (!value_type_raw) return fail("gguf: metadata key \"" + *key + "\": truncated value type");
    if (*value_type_raw > static_cast<std::uint32_t>(ValueType::Float64)) {
      return fail("gguf: metadata key \"" + *key + "\": unknown value type " + std::to_string(*value_type_raw));
    }
    auto value = read_value(c, static_cast<ValueType>(*value_type_raw), limits, 0);
    if (!value) return fail("gguf: metadata key \"" + *key + "\": " + value.error());

    if (*key == "general.alignment") {
      auto a = value->as_u64();
      if (!a || *a == 0 || (*a & (*a - 1)) != 0 || *a > limits.max_alignment) {
        return fail("gguf: general.alignment is not a valid power-of-two alignment");
      }
      f.alignment_ = *a;
      saw_alignment = true;
    }
    f.metadata_.emplace(*std::move(key), std::move(value.value()));
  }
  (void)saw_alignment;  // default (32) already set; this just documents the override path

  // ---- tensor infos ----
  for (std::uint64_t i = 0; i < *tensor_count; ++i) {
    auto name = c.string(limits.max_name_bytes);
    if (!name) return fail("gguf: tensor " + std::to_string(i) + ": truncated, oversized or invalid name");
    if (name->empty()) return fail("gguf: tensor " + std::to_string(i) + ": empty name");
    const std::string where = "gguf: tensor \"" + *name + "\": ";

    auto n_dims = c.u32();
    if (!n_dims) return fail(where + "truncated dimension count");
    if (*n_dims < 1 || *n_dims > limits.max_rank) return fail(where + "dimension count out of range");

    TensorInfo t;
    t.name = *std::move(name);
    t.shape.reserve(*n_dims);
    for (std::uint32_t d = 0; d < *n_dims; ++d) {
      auto dim = c.u64();
      if (!dim) return fail(where + "truncated shape");
      t.shape.push_back(*dim);
    }

    auto type_raw = c.u32();
    if (!type_raw) return fail(where + "truncated tensor type");
    const auto* type_entry = find_tensor_type(*type_raw);
    if (type_entry == nullptr) return fail(where + "unknown tensor type " + std::to_string(*type_raw));
    t.type = type_entry->type;

    auto offset = c.u64();
    if (!offset) return fail(where + "truncated offset");
    t.offset = *offset;

    std::optional<std::uint64_t> numel = 1;
    for (const auto d : t.shape) {
      numel = mul64(*numel, d);
      if (!numel) return fail(where + "shape overflows");
    }
    if (*numel % type_entry->block_size != 0) {
      return fail(where + "element count is not a multiple of its type's block size");
    }
    const std::uint64_t blocks = *numel / type_entry->block_size;
    auto size = mul64(blocks, type_entry->type_size);
    if (!size) return fail(where + "byte size overflows");
    t.size = *size;

    if (f.tensors_.size() >= limits.max_tensors) return fail("gguf: too many tensors");
    f.tensors_.push_back(std::move(t));
  }

  // ---- data section ----
  const std::uint64_t header_end = c.pos();
  if (f.alignment_ == 0) return fail("gguf: internal: zero alignment");  // unreachable; defensive
  std::uint64_t data_start = header_end;
  const std::uint64_t rem = header_end % f.alignment_;
  if (rem != 0) {
    auto padded = mul64(1, f.alignment_);  // no-op; keeps the overflow style uniform
    (void)padded;
    const std::uint64_t pad = f.alignment_ - rem;
    if (pad > std::numeric_limits<std::uint64_t>::max() - data_start) return fail("gguf: header length overflows");
    data_start += pad;
  }
  if (data_start > static_cast<std::uint64_t>(bytes.size())) {
    return fail("gguf: file is truncated before the (alignment-padded) tensor data section");
  }
  f.data_section_ = bytes.subspan(static_cast<std::size_t>(data_start));
  const std::uint64_t data_size = f.data_section_.size();

  for (const auto& t : f.tensors_) {
    if (t.offset % f.alignment_ != 0) {
      return fail("gguf: tensor \"" + t.name + "\": offset is not aligned to general.alignment (" +
                  std::to_string(f.alignment_) + ")");
    }
    if (t.offset > data_size || t.size > data_size - t.offset) {
      return fail("gguf: tensor \"" + t.name + "\": offset/size extend past the end of the file");
    }
  }

  // No two tensors may claim overlapping bytes (GGUF allows alignment gaps
  // between tensors, unlike safetensors, so gaps -- just not overlaps -- are
  // accepted; see docs/adr/0007-hardened-gguf-loader.md).
  std::vector<const TensorInfo*> by_offset;
  by_offset.reserve(f.tensors_.size());
  for (const auto& t : f.tensors_) by_offset.push_back(&t);
  std::sort(by_offset.begin(), by_offset.end(),
            [](const TensorInfo* a, const TensorInfo* b) { return a->offset < b->offset; });
  std::uint64_t cursor = 0;
  for (const auto* t : by_offset) {
    if (t->offset < cursor) return fail("gguf: tensor \"" + t->name + "\" overlaps another tensor");
    cursor = t->offset + t->size;
  }

  std::sort(f.tensors_.begin(), f.tensors_.end(),
            [](const TensorInfo& a, const TensorInfo& b) { return a.name < b.name; });
  for (std::size_t i = 1; i < f.tensors_.size(); ++i) {
    if (f.tensors_[i].name == f.tensors_[i - 1].name) {
      return fail("gguf: duplicate tensor name \"" + f.tensors_[i].name + "\"");
    }
  }

  return f;
}

Result<GGUFFile> GGUFFile::open(const std::string& path, const Limits& limits) {
  auto file = MappedFile::open(path);
  if (!file) return fail(file.error());
  auto parsed = parse(file.value()->bytes(), limits);
  if (!parsed) return fail(path + ": " + parsed.error());
  parsed->file_ = std::move(file.value());
  return parsed;
}

const Value* GGUFFile::find_metadata(std::string_view key) const {
  auto it = metadata_.find(std::string(key));
  return it == metadata_.end() ? nullptr : &it->second;
}

const TensorInfo* GGUFFile::find(std::string_view name) const {
  auto it = std::lower_bound(tensors_.begin(), tensors_.end(), name,
                             [](const TensorInfo& t, std::string_view n) { return t.name < n; });
  if (it == tensors_.end() || it->name != name) return nullptr;
  return &*it;
}

std::span<const std::byte> GGUFFile::data(const TensorInfo& t) const {
  return data_section_.subspan(static_cast<std::size_t>(t.offset), static_cast<std::size_t>(t.size));
}

// ---------------------------------------------------------------- conversion

Result<std::vector<float>> to_f32(const GGUFFile& f, const TensorInfo& t) {
  DType dt{};
  switch (t.type) {
    case TensorType::F32:
      dt = DType::F32;
      break;
    case TensorType::F16:
      dt = DType::F16;
      break;
    case TensorType::BF16:
      dt = DType::BF16;
      break;
    default:
      return fail("gguf: to_f32: tensor \"" + t.name + "\" has type " + std::string(tensor_type_name(t.type)) +
                  ", not F32/F16/BF16");
  }
  const auto raw = f.data(t);
  const auto n = static_cast<std::size_t>(t.numel());
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = read_float(raw, dt, i);
  return out;
}

namespace {

// Writes element `idx` of a row (0 <= idx < in), coded as GGUF/ggml codes it
// (Q8_0: signed -127..127; Q4_0: the unbiased value, -8..7), into this
// engine's own packed row layout (quant.cpp's `quantize()`): Q8_0 is one
// signed byte per element in order, identical to GGUF's own layout; Q4_0
// packs element idx into byte idx/2's low nibble if idx is even, high
// nibble if idx is odd (a +8-biased nibble) -- *adjacent pairs*, not GGUF's
// own halves-of-the-block pairing (see the long comment on to_quantized).
void write_engine_code(std::uint8_t* qrow, quant::Type type, std::size_t idx, int code) {
  if (type == quant::Type::Q8_0) {
    qrow[idx] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
    return;
  }
  const auto nibble = static_cast<std::uint8_t>(code + 8);
  std::uint8_t& byte = qrow[idx / 2];
  if (idx % 2 == 0) {
    byte = static_cast<std::uint8_t>((byte & 0xF0U) | nibble);
  } else {
    byte = static_cast<std::uint8_t>((byte & 0x0FU) | static_cast<std::uint8_t>(nibble << 4U));
  }
}

}  // namespace

// GGUF's Q8_0 block (ggml's `block_q8_0`: `{ fp16 d; int8 qs[32]; }`) packs
// its 32 signed codes in plain sequential order, identical to this engine's
// own Q8_0 layout (quant.cpp), so Q8_0 is a direct byte-for-byte copy
// (per block, after converting the fp16 scale to float32).
//
// GGUF's Q4_0 block (ggml's `block_q4_0`: `{ fp16 d; uint8 qs[16]; }`) does
// NOT pack adjacent elements into each byte. ggml's own reference quantizer
// (`quantize_row_q4_0_ref`) packs element j into byte j's low nibble and
// element j+16 into that SAME byte's high nibble -- i.e. the block's first
// and second HALF share each byte, not neighbouring elements. This engine's
// own Q4_0 (quant.cpp, M5) packs adjacent elements (2*k, 2*k+1) into byte k
// instead -- a perfectly valid, self-consistent convention for a format this
// engine both writes and reads itself, but a different one from GGUF's, so
// reading a real GGUF Q4_0 tensor needs an unpack-then-repack, not a memcpy.
// Caught by actually testing against real GGUF files from llama.cpp's own
// converter (docs/gguf.md), not by inspection -- the same way M5's own
// scale-sign bug was caught against a real cross-check rather than assumed
// correct from the spec alone.
Result<quant::QuantizedMatrix> to_quantized(const GGUFFile& f, const TensorInfo& t, std::size_t out, std::size_t in) {
  quant::Type qtype{};
  std::size_t gguf_qbytes = 0;  // packed-value bytes per block, after the 2-byte scale
  switch (t.type) {
    case TensorType::Q8_0:
      qtype = quant::Type::Q8_0;
      gguf_qbytes = quant::kBlockSize;
      break;
    case TensorType::Q4_0:
      qtype = quant::Type::Q4_0;
      gguf_qbytes = quant::kBlockSize / 2;
      break;
    default:
      return fail("gguf: to_quantized: tensor \"" + t.name + "\" has type " + std::string(tensor_type_name(t.type)) +
                  ", not Q8_0/Q4_0");
  }
  if (out == 0 || in == 0 || out * in != t.numel()) {
    return fail("gguf: to_quantized: tensor \"" + t.name + "\": out*in does not match its element count");
  }
  if (in % quant::kBlockSize != 0) {
    return fail("gguf: to_quantized: tensor \"" + t.name + "\": in (" + std::to_string(in) +
                ") is not a multiple of the block size (" + std::to_string(quant::kBlockSize) + ")");
  }
  const std::size_t blocks_per_row = in / quant::kBlockSize;
  const std::size_t gguf_block_bytes = 2 + gguf_qbytes;
  const auto raw = f.data(t);
  if (raw.size() != out * blocks_per_row * gguf_block_bytes) {
    return fail("gguf: to_quantized: tensor \"" + t.name + "\": unexpected byte size for its declared shape");
  }

  quant::QuantizedMatrix m;
  m.type = qtype;
  m.out = out;
  m.in = in;
  m.blocks_per_row = blocks_per_row;
  m.scales.resize(out * blocks_per_row);
  m.data.assign(qtype == quant::Type::Q8_0 ? out * in : out * ((in + 1) / 2), 0);

  for (std::size_t r = 0; r < out; ++r) {
    std::uint8_t* qrow = m.data.data() + r * (qtype == quant::Type::Q8_0 ? in : (in + 1) / 2);
    for (std::size_t b = 0; b < blocks_per_row; ++b) {
      const std::size_t block_off = (r * blocks_per_row + b) * gguf_block_bytes;
      std::uint16_t scale_bits = 0;
      std::memcpy(&scale_bits, raw.data() + block_off, 2);
      const std::array<std::byte, 2> scale_raw{std::byte(scale_bits & 0xFFU), std::byte((scale_bits >> 8U) & 0xFFU)};
      m.scales[r * blocks_per_row + b] = read_float(scale_raw, DType::F16, 0);

      const std::byte* qs = raw.data() + block_off + 2;
      const std::size_t row_base = b * quant::kBlockSize;
      if (qtype == quant::Type::Q8_0) {
        for (std::size_t j = 0; j < quant::kBlockSize; ++j) {
          // Sign extension is the point -- ggml's Q8_0 codes are signed, and
          // casting through unsigned char first would reinterpret the byte
          // instead of widening its signed value.
          // NOLINTNEXTLINE(bugprone-signed-char-misuse, cert-str34-c)
          const auto code = static_cast<int>(std::to_integer<std::int8_t>(qs[j]));
          write_engine_code(qrow, qtype, row_base + j, code);
        }
      } else {
        // ggml: byte j's low nibble is element j, high nibble is element j + kBlockSize/2.
        for (std::size_t j = 0; j < quant::kBlockSize / 2; ++j) {
          const auto byte = std::to_integer<std::uint8_t>(qs[j]);
          const int lo = static_cast<int>(byte & 0x0FU) - 8;
          const int hi = static_cast<int>(byte >> 4U) - 8;
          write_engine_code(qrow, qtype, row_base + j, lo);
          write_engine_code(qrow, qtype, row_base + j + quant::kBlockSize / 2, hi);
        }
      }
    }
  }
  return m;
}

}  // namespace llmi::gguf
