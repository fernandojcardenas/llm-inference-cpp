#include "llmi/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <mutex>
#include <queue>

#include "llmi/model/model.hpp"
#include "llmi/util/json.hpp"
#include "llmi/util/utf8.hpp"

namespace llmi {

// ---------------------------------------------------------------- Unicode classes

namespace {

struct CodeRange {
  char32_t first;
  char32_t last;
};

#include "unicode_tables.inc"

template <std::size_t N>
bool in_ranges(const CodeRange (&ranges)[N], char32_t cp) {
  const auto* it = std::upper_bound(std::begin(ranges), std::end(ranges), cp,
                                    [](char32_t c, const CodeRange& r) { return c < r.first; });
  if (it == std::begin(ranges)) return false;
  --it;
  return cp <= it->last;
}

}  // namespace

namespace unicode {

bool is_letter(char32_t cp) { return in_ranges(kLetterRanges, cp); }
bool is_number(char32_t cp) { return in_ranges(kNumberRanges, cp); }
bool is_numeric(char32_t cp) { return in_ranges(kNumericRanges, cp); }

// The Unicode White_Space property (PropList.txt), which regex \s matches.
bool is_whitespace(char32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
         cp == 0x3000;
}

const char* version() { return kUnicodeVersion; }
const char* digits_version() { return kDigitsUnicodeVersion; }

}  // namespace unicode

// ---------------------------------------------------------------- cache

struct Tokenizer::Cache {
  static constexpr std::size_t kMaxEntries = 100'000;
  std::mutex mu;
  struct Entry {
    std::vector<TokenId> ids;
    std::size_t dropped = 0;
  };
  std::unordered_map<std::string, Entry> words;
};

// ---------------------------------------------------------------- loading

namespace {

bool null_or_empty(const json::Value* v) {
  return v == nullptr || v->is_null() || (v->is_string() && v->text().empty());
}

bool is_false_or_null(const json::Value* v) {
  return v == nullptr || v->is_null() || (v->is_bool() && !v->as_bool());
}

// GPT-2's reversible map from bytes to printable characters.
std::array<char32_t, 256> bytes_to_unicode() {
  std::array<char32_t, 256> map{};
  std::array<bool, 256> printable{};
  for (unsigned b = '!'; b <= '~'; ++b) printable[b] = true;
  for (unsigned b = 0xA1; b <= 0xAC; ++b) printable[b] = true;
  for (unsigned b = 0xAE; b <= 0xFF; ++b) printable[b] = true;
  char32_t next = 256;
  for (unsigned b = 0; b < 256; ++b) map[b] = printable[b] ? static_cast<char32_t>(b) : next++;
  return map;
}

Result<bool> check_byte_level(const json::Value& v, std::string_view where) {
  if (!v.is_object()) return fail(std::string(where) + " is not an object");
  if (!is_false_or_null(v.find("add_prefix_space"))) {
    return fail(std::string(where) + ": add_prefix_space is not supported");
  }
  const json::Value* use_regex = v.find("use_regex");
  return use_regex == nullptr || (use_regex->is_bool() && use_regex->as_bool());
}

}  // namespace

Result<Tokenizer> Tokenizer::from_json(std::string_view json_text) {
  auto root = json::parse(json_text);
  if (!root) return fail("tokenizer: " + root.error());
  if (!root->is_object()) return fail("tokenizer: not a JSON object");

  const json::Value* normalizer = root->find("normalizer");
  if (normalizer != nullptr && !normalizer->is_null()) return fail("tokenizer: normalizers are not supported");

  Tokenizer t;

  // Pre-tokenizer: ByteLevel with the GPT-2 pattern, optionally after Digits.
  const json::Value* pre = root->find("pre_tokenizer");
  if (pre == nullptr || !pre->is_object()) return fail("tokenizer: missing pre_tokenizer");
  const json::Value* pre_type = pre->find("type");
  if (pre_type == nullptr || !pre_type->is_string()) return fail("tokenizer: pre_tokenizer has no type");
  const json::Value* byte_level = nullptr;
  if (pre_type->text() == "ByteLevel") {
    byte_level = pre;
  } else if (pre_type->text() == "Sequence") {
    const json::Value* seq = pre->find("pretokenizers");
    if (seq == nullptr || !seq->is_array()) return fail("tokenizer: Sequence without pretokenizers");
    for (std::size_t i = 0; i < seq->items().size(); ++i) {
      const json::Value& p = seq->items()[i];
      const json::Value* pt = p.find("type");
      if (pt == nullptr || !pt->is_string()) return fail("tokenizer: pre-tokenizer without a type");
      const bool last = i + 1 == seq->items().size();
      if (pt->text() == "Digits" && !last && byte_level == nullptr && !t.split_digits_) {
        t.split_digits_ = true;
        const json::Value* ind = p.find("individual_digits");
        t.individual_digits_ = ind != nullptr && ind->is_bool() && ind->as_bool();
      } else if (pt->text() == "ByteLevel" && last) {
        byte_level = &p;
      } else {
        return fail("tokenizer: unsupported pre-tokenizer sequence (supported: [Digits], ByteLevel)");
      }
    }
  } else {
    return fail("tokenizer: unsupported pre-tokenizer " + pre_type->text());
  }
  if (byte_level == nullptr) return fail("tokenizer: no ByteLevel pre-tokenizer");
  auto regex = check_byte_level(*byte_level, "tokenizer: ByteLevel pre-tokenizer");
  if (!regex) return fail(regex.error());
  if (!regex.value()) return fail("tokenizer: ByteLevel without the GPT-2 pattern is not supported");

  const json::Value* decoder = root->find("decoder");
  if (decoder == nullptr || !decoder->is_object() || decoder->find("type") == nullptr ||
      decoder->find("type")->text() != "ByteLevel") {
    return fail("tokenizer: only the ByteLevel decoder is supported");
  }
  const json::Value* post = root->find("post_processor");
  const bool post_ok = post == nullptr || post->is_null() ||
                       (post->is_object() && post->find("type") != nullptr && post->find("type")->text() == "ByteLevel");
  if (!post_ok) {
    return fail("tokenizer: unsupported post_processor (template processors add tokens; not supported yet)");
  }

  // Model: BPE.
  const json::Value* model = root->find("model");
  if (model == nullptr || !model->is_object()) return fail("tokenizer: missing model");
  const json::Value* type = model->find("type");
  if (type == nullptr || !type->is_string() || type->text() != "BPE") return fail("tokenizer: only BPE models are supported");
  if (!null_or_empty(model->find("dropout"))) return fail("tokenizer: BPE dropout is not supported");
  if (!null_or_empty(model->find("unk_token"))) return fail("tokenizer: unk_token is not supported");
  if (!null_or_empty(model->find("continuing_subword_prefix")) || !null_or_empty(model->find("end_of_word_suffix"))) {
    return fail("tokenizer: subword prefixes/suffixes are not supported");
  }
  if (!is_false_or_null(model->find("byte_fallback"))) return fail("tokenizer: byte_fallback is not supported");
  if (const json::Value* im = model->find("ignore_merges"); im != nullptr && im->is_bool()) t.ignore_merges_ = im->as_bool();

  const json::Value* vocab = model->find("vocab");
  if (vocab == nullptr || !vocab->is_object()) return fail("tokenizer: missing vocab");
  constexpr std::uint64_t kMaxVocab = 1U << 22U;
  auto set_token = [&t](TokenId id, const std::string& s) -> bool {
    const auto idx = static_cast<std::size_t>(id);
    if (idx >= t.id_to_token_.size()) {
      t.id_to_token_.resize(idx + 1);
      t.special_.resize(idx + 1, false);
      t.added_.resize(idx + 1, false);
    }
    if (!t.id_to_token_[idx].empty() && t.id_to_token_[idx] != s) return false;
    t.id_to_token_[idx] = s;
    return true;
  };
  for (const auto& [tok, idv] : vocab->members()) {
    auto id = idv.as_u64();
    if (!id || *id >= kMaxVocab) return fail("tokenizer: bad id for token " + tok);
    if (tok.empty() || !set_token(static_cast<TokenId>(*id), tok)) return fail("tokenizer: duplicate id " + std::to_string(*id));
    t.vocab_.emplace(tok, static_cast<TokenId>(*id));
  }

  // Byte-level alphabet. A byte without a token is recorded, not fatal:
  // the reference implementation loads such vocabularies and drops the byte.
  t.byte_to_char_ = bytes_to_unicode();
  for (unsigned b = 0; b < 256; ++b) {
    std::string s;
    utf8::append(s, t.byte_to_char_[b]);
    auto it = t.vocab_.find(s);
    if (it == t.vocab_.end()) {
      t.byte_token_[b] = -1;
      t.missing_bytes_.push_back(static_cast<std::uint8_t>(b));
    } else {
      t.byte_token_[b] = it->second;
    }
    t.char_to_byte_.emplace(t.byte_to_char_[b], static_cast<std::uint8_t>(b));
  }
  if (t.missing_bytes_.size() > 64) return fail("tokenizer: vocab lacks most byte tokens; not a byte-level BPE vocab");

  const json::Value* merges = model->find("merges");
  if (merges == nullptr || !merges->is_array()) return fail("tokenizer: missing merges");
  std::uint32_t rank = 0;
  for (const auto& m : merges->items()) {
    std::string a;
    std::string b;
    if (m.is_string()) {
      const auto sp = m.text().find(' ');
      if (sp == std::string::npos || m.text().find(' ', sp + 1) != std::string::npos) {
        return fail("tokenizer: merge \"" + m.text() + "\" is not two tokens");
      }
      a = m.text().substr(0, sp);
      b = m.text().substr(sp + 1);
    } else if (m.is_array() && m.items().size() == 2 && m.items()[0].is_string() && m.items()[1].is_string()) {
      a = m.items()[0].text();
      b = m.items()[1].text();
    } else {
      return fail("tokenizer: malformed merge");
    }
    const TokenId ia = t.id(a);
    const TokenId ib = t.id(b);
    const TokenId iab = t.id(a + b);
    if (ia < 0 || ib < 0 || iab < 0) {
      std::string msg = "tokenizer: merge \"";
      msg += a;
      msg += ' ';
      msg += b;
      msg += "\" uses a token not in the vocab";
      return fail(msg);
    }
    const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ia)) << 32U) |
                              static_cast<std::uint32_t>(ib);
    t.merges_.emplace(key, MergeTarget{rank++, iab});  // first (lowest-rank) occurrence wins
  }

  // Added tokens.
  if (const json::Value* added = root->find("added_tokens"); added != nullptr && !added->is_null()) {
    if (!added->is_array()) return fail("tokenizer: added_tokens is not an array");
    for (const auto& a : added->items()) {
      const json::Value* id = a.find("id");
      const json::Value* content = a.find("content");
      if (id == nullptr || content == nullptr || !content->is_string() || content->text().empty()) {
        return fail("tokenizer: malformed added token");
      }
      const std::uint64_t idn = id->as_u64().value_or(kMaxVocab);
      if (idn >= kMaxVocab) return fail("tokenizer: bad added-token id");
      for (const char* flag : {"single_word", "lstrip", "rstrip"}) {
        if (!is_false_or_null(a.find(flag))) return fail(std::string("tokenizer: added-token option ") + flag + " is not supported");
      }
      const json::Value* sp = a.find("special");
      AddedToken tok{content->text(), static_cast<TokenId>(idn), sp != nullptr && sp->is_bool() && sp->as_bool()};
      if (!set_token(tok.id, tok.content)) return fail("tokenizer: added token id clashes with the vocab");
      t.vocab_.emplace(tok.content, tok.id);
      t.added_[static_cast<std::size_t>(tok.id)] = true;
      t.special_[static_cast<std::size_t>(tok.id)] = tok.special;
      t.added_tokens_.push_back(std::move(tok));
    }
    std::stable_sort(t.added_tokens_.begin(), t.added_tokens_.end(),
                     [](const AddedToken& x, const AddedToken& y) { return x.content.size() > y.content.size(); });
  }
  for (std::size_t i = 0; i < t.id_to_token_.size(); ++i) {
    if (t.id_to_token_[i].empty()) return fail("tokenizer: no token has id " + std::to_string(i));
  }
  t.cache_ = std::make_shared<Cache>();
  return t;
}

Result<Tokenizer> Tokenizer::load(const std::string& path) {
  auto text = read_file(path, 64U << 20U);
  if (!text) return fail(text.error());
  auto t = from_json(text.value());
  if (!t) return fail(path + ": " + t.error());
  return t;
}

// ---------------------------------------------------------------- lookups

std::string_view Tokenizer::token(TokenId id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) return {};
  return id_to_token_[static_cast<std::size_t>(id)];
}

TokenId Tokenizer::id(std::string_view token) const {
  auto it = vocab_.find(std::string(token));
  return it == vocab_.end() ? -1 : it->second;
}

bool Tokenizer::is_special(TokenId id) const {
  return id >= 0 && static_cast<std::size_t>(id) < special_.size() && special_[static_cast<std::size_t>(id)];
}

// ---------------------------------------------------------------- BPE

void Tokenizer::encode_word(std::string_view word, std::vector<TokenId>& out, std::size_t& dropped) const {
  {
    const std::lock_guard lock(cache_->mu);
    auto it = cache_->words.find(std::string(word));
    if (it != cache_->words.end()) {
      out.insert(out.end(), it->second.ids.begin(), it->second.ids.end());
      dropped += it->second.dropped;
      return;
    }
  }

  std::vector<TokenId> result;
  std::size_t word_dropped = 0;
  bool done = false;
  if (ignore_merges_) {
    std::string mapped;
    for (const char c : word) utf8::append(mapped, byte_to_char_[static_cast<unsigned char>(c)]);
    auto it = vocab_.find(mapped);
    if (it != vocab_.end()) {
      result.push_back(it->second);
      done = true;
    }
  }

  if (!done) {
    // Symbols as a doubly linked list over a vector; merges come off a
    // min-heap ordered by (rank, position), as in Hugging Face tokenizers.
    struct Symbol {
      TokenId id;
      std::int32_t prev;
      std::int32_t next;
      bool alive;
    };
    // Bytes without a token are skipped (so their neighbours can merge
    // across the gap), exactly as the reference implementation does.
    std::vector<Symbol> sym;
    sym.reserve(word.size());
    for (const char c : word) {
      const TokenId tid = byte_token_[static_cast<unsigned char>(c)];
      if (tid < 0) {
        ++word_dropped;
        continue;
      }
      const auto pos = static_cast<std::int32_t>(sym.size());
      if (!sym.empty()) sym.back().next = pos;
      sym.push_back({tid, pos - 1, -1, true});
    }
    struct Cand {
      std::uint32_t rank;
      std::int32_t pos;
      TokenId left;
      TokenId right;
      TokenId merged;
      bool operator>(const Cand& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
    };
    std::priority_queue<Cand, std::vector<Cand>, std::greater<>> heap;
    auto push = [&](std::int32_t pos) {
      if (pos < 0) return;
      const std::int32_t nxt = sym[static_cast<std::size_t>(pos)].next;
      if (nxt < 0) return;
      const TokenId l = sym[static_cast<std::size_t>(pos)].id;
      const TokenId r = sym[static_cast<std::size_t>(nxt)].id;
      const std::uint64_t key =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(l)) << 32U) | static_cast<std::uint32_t>(r);
      auto it = merges_.find(key);
      if (it != merges_.end()) heap.push({it->second.rank, pos, l, r, it->second.merged});
    };
    for (std::size_t i = 0; i + 1 < sym.size(); ++i) push(static_cast<std::int32_t>(i));
    while (!heap.empty()) {
      const Cand c = heap.top();
      heap.pop();
      Symbol& s = sym[static_cast<std::size_t>(c.pos)];
      if (!s.alive || s.id != c.left || s.next < 0) continue;  // stale
      Symbol& n = sym[static_cast<std::size_t>(s.next)];
      if (n.id != c.right) continue;  // stale
      s.id = c.merged;
      n.alive = false;
      s.next = n.next;
      if (s.next >= 0) sym[static_cast<std::size_t>(s.next)].prev = c.pos;
      push(s.prev);
      push(c.pos);
    }
    for (std::int32_t i = sym.empty() ? -1 : 0; i >= 0; i = sym[static_cast<std::size_t>(i)].next) {
      result.push_back(sym[static_cast<std::size_t>(i)].id);
    }
  }

  out.insert(out.end(), result.begin(), result.end());
  dropped += word_dropped;
  const std::lock_guard lock(cache_->mu);
  if (cache_->words.size() >= Cache::kMaxEntries) cache_->words.clear();
  cache_->words.emplace(std::string(word), Cache::Entry{std::move(result), word_dropped});
}

// ---------------------------------------------------------------- pre-tokenizer

namespace {

using unicode::is_letter;
using unicode::is_number;
using unicode::is_whitespace;

bool is_other(char32_t c) { return !is_whitespace(c) && !is_letter(c) && !is_number(c); }

// Length of the GPT-2 pattern's match at cps[i] within [i, end); always >= 1.
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::size_t gpt2_match(const std::vector<char32_t>& cps, std::size_t i, std::size_t end) {
  const std::size_t avail = end - i;
  if (cps[i] == U'\'' && avail >= 2) {
    const char32_t c1 = cps[i + 1];
    if (c1 == U's' || c1 == U't' || c1 == U'm' || c1 == U'd') return 2;
    if (avail >= 3) {
      const char32_t c2 = cps[i + 2];
      if ((c1 == U'r' && c2 == U'e') || (c1 == U'v' && c2 == U'e') || (c1 == U'l' && c2 == U'l')) return 3;
    }
  }
  const std::size_t start = (cps[i] == U' ' && avail >= 2) ? i + 1 : i;
  for (auto cls : {is_letter, is_number, is_other}) {
    if (start < end && cls(cps[start])) {
      std::size_t j = start + 1;
      while (j < end && cls(cps[j])) ++j;
      return j - i;
    }
  }
  // Whitespace run (cps[i] is whitespace here, possibly a lone space before
  // a character no earlier branch took).
  std::size_t j = i;
  while (j < end && is_whitespace(cps[j])) ++j;
  const std::size_t run = j - i;
  if (j == end || run == 1) return run;  // \s+(?!\S) at the end, or \s+
  return run - 1;                          // leave one space to prefix the next word
}

}  // namespace

void Tokenizer::encode_text(const std::vector<char32_t>& cps, std::size_t begin, std::size_t end,
                            std::vector<TokenId>& out, std::size_t& dropped) const {
  std::string word;
  auto run_pattern = [&](std::size_t b, std::size_t e) {
    std::size_t i = b;
    while (i < e) {
      const std::size_t len = gpt2_match(cps, i, e);
      word.clear();
      for (std::size_t k = i; k < i + len; ++k) utf8::append(word, cps[k]);
      encode_word(word, out, dropped);
      i += len;
    }
  };
  if (!split_digits_) {
    run_pattern(begin, end);
    return;
  }
  std::size_t piece = begin;
  std::size_t i = begin;
  while (i < end) {
    if (!unicode::is_numeric(cps[i])) {
      ++i;
      continue;
    }
    run_pattern(piece, i);
    std::size_t j = i + 1;
    if (!individual_digits_) {
      while (j < end && unicode::is_numeric(cps[j])) ++j;
    }
    run_pattern(i, j);
    piece = i = j;
  }
  run_pattern(piece, end);
}

Result<std::vector<TokenId>> Tokenizer::encode(std::string_view text, bool parse_special, std::size_t* dropped) const {
  auto cps_opt = utf8::decode(text);
  if (!cps_opt) return fail("tokenizer: input is not valid UTF-8");
  const std::vector<char32_t>& cps = *cps_opt;
  std::vector<TokenId> out;
  std::size_t lost = 0;
  if (dropped != nullptr) *dropped = 0;

  if (!parse_special || added_tokens_.empty()) {
    encode_text(cps, 0, cps.size(), out, lost);
    if (dropped != nullptr) *dropped = lost;
    return out;
  }
  // Byte offset of each code point, to match added tokens on bytes.
  std::vector<std::size_t> byte_at(cps.size() + 1);
  {
    std::size_t b = 0;
    for (std::size_t k = 0; k < cps.size(); ++k) {
      byte_at[k] = b;
      std::string tmp;
      utf8::append(tmp, cps[k]);
      b += tmp.size();
    }
    byte_at[cps.size()] = b;
  }
  std::size_t seg = 0;
  std::size_t k = 0;
  while (k < cps.size()) {
    const AddedToken* hit = nullptr;
    for (const auto& a : added_tokens_) {  // longest first
      if (text.substr(byte_at[k], a.content.size()) == a.content) {
        hit = &a;
        break;
      }
    }
    if (hit == nullptr) {
      ++k;
      continue;
    }
    encode_text(cps, seg, k, out, lost);
    out.push_back(hit->id);
    const std::size_t end_byte = byte_at[k] + hit->content.size();
    while (k < cps.size() && byte_at[k] < end_byte) ++k;
    seg = k;
  }
  encode_text(cps, seg, cps.size(), out, lost);
  if (dropped != nullptr) *dropped = lost;
  return out;
}

Result<std::string> Tokenizer::decode(const std::vector<TokenId>& ids, bool skip_special) const {
  std::string out;
  for (const TokenId id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
      return fail("tokenizer: token id " + std::to_string(id) + " out of range");
    }
    const auto idx = static_cast<std::size_t>(id);
    if (added_[idx]) {
      if (!(skip_special && special_[idx])) out += id_to_token_[idx];
      continue;
    }
    const std::string& tok = id_to_token_[idx];
    std::size_t i = 0;
    while (i < tok.size()) {
      auto cp = utf8::next(tok, i);
      if (!cp) return fail("tokenizer: vocab entry is not valid UTF-8");
      auto it = char_to_byte_.find(*cp);
      if (it == char_to_byte_.end()) {
        utf8::append(out, *cp);  // not a byte-level character; keep as text
      } else {
        out.push_back(static_cast<char>(it->second));
      }
    }
  }
  return out;
}

}  // namespace llmi
