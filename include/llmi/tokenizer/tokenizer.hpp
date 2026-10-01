#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "llmi/util/result.hpp"

namespace llmi {

using TokenId = std::int32_t;

// A byte-level BPE tokenizer read from a Hugging Face tokenizer.json, the
// scheme GPT-2, SmolLM2, Llama 3 and Qwen use. Encoding:
//   1. split out added tokens (<|im_start|> ...), leftmost-longest match;
//   2. pre-tokenize: optionally isolate digits, then split with the GPT-2
//      pattern ('s|'t|... | ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+);
//   3. map each piece's UTF-8 bytes to printable characters (GPT-2's
//      bytes_to_unicode) and apply the BPE merges, lowest rank first.
// Only the features these models use are implemented; anything else in the
// file (normalizers, other pre-tokenizers, byte fallback, dropout) is
// rejected at load time instead of being silently ignored.
class Tokenizer {
 public:
  static Result<Tokenizer> from_json(std::string_view json_text);
  static Result<Tokenizer> load(const std::string& path);

  // Input must be valid UTF-8 (anything else is an error, not a guess).
  // With parse_special=false, added tokens in the text are treated as plain
  // text (useful when the text is untrusted user input).
  //
  // Some vocabularies lack a token for a few bytes (SmolLM2 has none for 21
  // bytes, six of them ASCII control characters). Like the reference
  // implementation, encoding drops those bytes; `dropped`, if given, receives
  // how many were dropped so callers can refuse or warn instead.
  [[nodiscard]] Result<std::vector<TokenId>> encode(std::string_view text, bool parse_special = true,
                                                    std::size_t* dropped = nullptr) const;

  // Bytes with no token in the vocabulary (dropped by encode).
  [[nodiscard]] const std::vector<std::uint8_t>& missing_bytes() const { return missing_bytes_; }

  // Raw bytes; a prefix of a sequence may end inside a UTF-8 character.
  [[nodiscard]] Result<std::string> decode(const std::vector<TokenId>& ids, bool skip_special = false) const;

  [[nodiscard]] std::size_t vocab_size() const { return id_to_token_.size(); }
  [[nodiscard]] std::string_view token(TokenId id) const;
  [[nodiscard]] TokenId id(std::string_view token) const;  // -1 if absent
  [[nodiscard]] bool is_special(TokenId id) const;

 private:
  struct AddedToken {
    std::string content;
    TokenId id = -1;
    bool special = false;
  };
  struct MergeTarget {
    std::uint32_t rank = 0;
    TokenId merged = -1;
  };

  void encode_word(std::string_view utf8_word, std::vector<TokenId>& out, std::size_t& dropped) const;
  void encode_text(const std::vector<char32_t>& cps, std::size_t begin, std::size_t end, std::vector<TokenId>& out,
                   std::size_t& dropped) const;

  std::unordered_map<std::string, TokenId> vocab_;
  std::vector<std::string> id_to_token_;
  std::vector<bool> special_;
  std::vector<bool> added_;
  std::unordered_map<std::uint64_t, MergeTarget> merges_;
  std::vector<AddedToken> added_tokens_;  // sorted longest first
  std::array<TokenId, 256> byte_token_{};  // -1: no token for this byte
  std::vector<std::uint8_t> missing_bytes_;
  std::array<char32_t, 256> byte_to_char_{};
  std::unordered_map<char32_t, std::uint8_t> char_to_byte_;
  bool split_digits_ = false;
  bool individual_digits_ = false;
  bool ignore_merges_ = false;

  // Per-word cache; words repeat constantly in real text.
  struct Cache;
  std::shared_ptr<Cache> cache_;
};

// Character classes used by pre-tokenization. The reference implementation
// uses two Unicode versions: its regex engine's for \p{L} and \p{N}, and
// Rust's char::is_numeric (newer) for the Digits pre-tokenizer.
namespace unicode {
bool is_letter(char32_t cp);      // \p{L}, Unicode version()
bool is_number(char32_t cp);      // \p{N}, Unicode version()
bool is_numeric(char32_t cp);     // Digits pre-tokenizer, Unicode digits_version()
bool is_whitespace(char32_t cp);  // \s: the White_Space property
const char* version();
const char* digits_version();
}  // namespace unicode

}  // namespace llmi
