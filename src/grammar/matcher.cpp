#include "llmi/grammar/matcher.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "llmi/util/utf8.hpp"

namespace llmi::grammar {

namespace {

constexpr char32_t kMaxCodePoint = 0x10FFFF;

// Total length of the UTF-8 sequence that starts with `lead`, or 0 if `lead`
// cannot start one (a continuation byte, an overlong 0xC0/0xC1, or >= 0xF5).
std::size_t sequence_length(std::uint8_t lead) {
  if (lead < 0x80) return 1;
  if (lead >= 0xC2 && lead <= 0xDF) return 2;
  if (lead >= 0xE0 && lead <= 0xEF) return 3;
  if (lead >= 0xF0 && lead <= 0xF4) return 4;
  return 0;
}

// The payload bits of a lead byte, and the smallest code point a sequence of
// that length may encode (anything smaller would be an overlong encoding).
std::uint32_t lead_payload_mask(std::size_t length) {
  switch (length) {
    case 2: return 0x1FU;
    case 3: return 0x0FU;
    default: return 0x07U;
  }
}

char32_t smallest_code_point(std::size_t length) {
  switch (length) {
    case 2: return 0x80;
    case 3: return 0x800;
    default: return 0x10000;
  }
}

void dedupe(std::vector<Matcher::Stack>& stacks) {
  std::sort(stacks.begin(), stacks.end());
  stacks.erase(std::unique(stacks.begin(), stacks.end()), stacks.end());
}

}  // namespace

Matcher::Matcher(std::shared_ptr<const Grammar> grammar, Limits limits)
    : grammar_(std::move(grammar)), limits_(limits) {
  const std::uint32_t root = grammar_->root();
  const Rule& r = grammar_->rules()[root];
  std::vector<Stack> stacks;
  for (std::uint32_t a = 0; a < r.alternatives.size(); ++a) {
    Stack s;
    if (!r.alternatives[a].empty()) s.push_back(Pos{root, a, 0});
    if (!normalize(std::move(s), stacks)) {
      stacks.clear();  // over a limit: fail closed, nothing is accepted
      break;
    }
  }
  dedupe(stacks);
  if (stacks.size() > limits_.max_stacks) stacks.clear();
  state_.stacks = std::move(stacks);
}

const Symbol& Matcher::symbol_at(const Pos& p) const {
  return grammar_->rules()[p.rule].alternatives[p.alt][p.idx];
}

// Expands rule references at the top of `start` until every resulting stack
// has a character set on top (or is empty: a complete parse), appending them
// to `out`. The grammar has no left recursion, so this terminates; the
// limits only bound how much work an adversarial grammar can cause.
bool Matcher::normalize(Stack start, std::vector<Stack>& out) const {
  std::vector<Stack> work;
  work.push_back(std::move(start));
  std::size_t expansions = 0;
  while (!work.empty()) {
    Stack s = std::move(work.back());
    work.pop_back();
    if (s.empty()) {
      out.push_back(std::move(s));
      continue;
    }
    if (s.size() > limits_.max_stack_depth) return false;
    const Pos top = s.back();
    const Symbol& sym = symbol_at(top);
    if (sym.kind == Symbol::Kind::Chars) {
      out.push_back(std::move(s));
      if (out.size() > limits_.max_stacks) return false;
      continue;
    }
    if (++expansions > limits_.max_expansions) return false;
    s.pop_back();
    const Sequence& seq = grammar_->rules()[top.rule].alternatives[top.alt];
    if (top.idx + 1 < seq.size()) s.push_back(Pos{top.rule, top.alt, top.idx + 1});
    const Rule& callee = grammar_->rules()[sym.rule];
    for (std::uint32_t a = 0; a < callee.alternatives.size(); ++a) {
      Stack t = s;
      if (!callee.alternatives[a].empty()) t.push_back(Pos{sym.rule, a, 0});
      work.push_back(std::move(t));
    }
  }
  return true;
}

bool Matcher::advance(const std::vector<Stack>& in, char32_t cp, std::vector<Stack>& out) const {
  out.clear();
  for (const Stack& s : in) {
    if (s.empty()) continue;
    const Pos top = s.back();
    if (!symbol_at(top).chars.contains(cp)) continue;
    Stack t = s;
    t.pop_back();
    const Sequence& seq = grammar_->rules()[top.rule].alternatives[top.alt];
    if (top.idx + 1 < seq.size()) t.push_back(Pos{top.rule, top.alt, top.idx + 1});
    if (!normalize(std::move(t), out)) return false;
  }
  dedupe(out);
  return out.size() <= limits_.max_stacks;
}

bool Matcher::viable(const std::vector<Stack>& stacks, char32_t lo, char32_t hi) const {
  return std::any_of(stacks.begin(), stacks.end(),
                     [&](const Stack& s) { return !s.empty() && symbol_at(s.back()).chars.intersects(lo, hi); });
}

bool Matcher::step(const State& in, std::uint8_t byte, State& out) const {
  if (in.partial_len == 0) {
    const std::size_t len = sequence_length(byte);
    if (len == 0) return false;
    if (len == 1) {
      out.partial_len = 0;
      return advance(in.stacks, byte, out.stacks) && !out.stacks.empty();
    }
    out.stacks = in.stacks;
    out.partial = {byte, 0, 0, 0};
    out.partial_len = 1;
  } else {
    if (byte < 0x80 || byte > 0xBF) return false;  // not a continuation byte
    out.partial = in.partial;
    out.partial.at(in.partial_len) = byte;
    out.partial_len = static_cast<std::uint8_t>(in.partial_len + 1);
    const std::size_t need = sequence_length(out.partial[0]);
    if (out.partial_len == need) {
      const std::string_view bytes(reinterpret_cast<const char*>(out.partial.data()), need);
      std::size_t i = 0;
      const auto cp = utf8::next(bytes, i);  // strict: rejects overlongs and surrogates
      if (!cp) return false;
      out.partial_len = 0;
      return advance(in.stacks, *cp, out.stacks) && !out.stacks.empty();
    }
    out.stacks = in.stacks;
  }

  // Half a character: allowed if some stack could match a character that
  // starts with these bytes. [lo, hi] = every code point they could become.
  const std::size_t need = sequence_length(out.partial[0]);
  const std::size_t have = out.partial_len;
  if (need < 2 || have >= need) return false;  // cannot happen; keeps the shifts below in range
  std::uint32_t v = out.partial[0] & lead_payload_mask(need);
  for (std::size_t k = 1; k < have; ++k) v = (v << 6U) | (out.partial.at(k) & 0x3FU);
  const auto missing_bits = static_cast<unsigned>(6 * (need - have));  // 6, 12 or 18
  char32_t lo = v << missing_bits;
  char32_t hi = lo | ((1U << missing_bits) - 1U);
  lo = std::max(lo, smallest_code_point(need));
  hi = std::min(hi, kMaxCodePoint);
  return lo <= hi && viable(out.stacks, lo, hi);
}

bool Matcher::accept(std::string_view bytes) {
  State cur = state_;
  for (const char c : bytes) {
    State next;
    if (!step(cur, static_cast<std::uint8_t>(c), next)) return false;
    cur = std::move(next);
  }
  state_ = std::move(cur);
  return true;
}

bool Matcher::can_accept(std::string_view bytes) const {
  State cur = state_;
  for (const char c : bytes) {
    State next;
    if (!step(cur, static_cast<std::uint8_t>(c), next)) return false;
    cur = std::move(next);
  }
  return true;
}

bool Matcher::complete() const {
  return state_.partial_len == 0 &&
         std::any_of(state_.stacks.begin(), state_.stacks.end(), [](const Stack& s) { return s.empty(); });
}

bool Matcher::has_continuation() const {
  return state_.partial_len != 0 ||
         std::any_of(state_.stacks.begin(), state_.stacks.end(), [](const Stack& s) { return !s.empty(); });
}

void Matcher::allowed_tokens_direct(const TokenTrie& trie, std::vector<std::uint8_t>& mask) const {
  mask.assign(trie.vocab_size(), 0);
  last_nodes_visited_ = 0;
  if (trie.nodes_.empty() || dead()) return;
  // Depth-first over the trie, carrying the matcher state reached by each
  // node's bytes; a byte the grammar rejects prunes that whole subtree.
  std::vector<std::pair<std::uint32_t, State>> work;
  work.emplace_back(0U, state_);
  while (!work.empty()) {
    auto [node, st] = std::move(work.back());
    work.pop_back();
    ++last_nodes_visited_;
    const TokenTrie::Node& n = trie.nodes_[node];
    for (const std::int32_t id : n.tokens) mask[static_cast<std::size_t>(id)] = 1;
    for (const auto& [byte, child] : n.children) {
      State next;
      if (step(st, byte, next)) work.emplace_back(child, std::move(next));
    }
  }
}

namespace {

// Two matcher states are equal when they would accept exactly the same bytes.
// States are kept canonical (stacks sorted and deduplicated by normalize/the
// constructor), so equal states have identical `stacks`; only the live bytes
// of a half-read character matter.
bool states_equal(const Matcher::State& a, const Matcher::State& b) {
  if (a.partial_len != b.partial_len) return false;
  for (std::uint8_t i = 0; i < a.partial_len; ++i) {
    if (a.partial.at(i) != b.partial.at(i)) return false;
  }
  return a.stacks == b.stacks;
}

}  // namespace

void Matcher::allowed_tokens(const TokenTrie& trie, std::vector<std::uint8_t>& mask) const {
  // Same depth-first walk as allowed_tokens_direct, but the matcher state at
  // each node is interned to an index and every (state, byte) transition is
  // computed once and cached. A trie subtree reached under one state -- a
  // free-form string allows almost the whole vocabulary -- then costs a cache
  // lookup per node instead of rebuilding the state (and its allocations) at
  // every node. The result is identical to allowed_tokens_direct().
  constexpr std::size_t kMaxMemoStates = 2048;  // beyond this, fall back
  constexpr std::int32_t kUnknown = -2;
  constexpr std::int32_t kReject = -1;
  constexpr std::int32_t kMemoFull = -3;

  mask.assign(trie.vocab_size(), 0);
  last_nodes_visited_ = 0;
  if (trie.nodes_.empty() || dead()) return;

  std::vector<State> states;
  states.push_back(state_);
  std::vector<std::array<std::int32_t, 256>> trans;  // per state: byte -> child index / kReject / kUnknown
  trans.emplace_back();
  trans.back().fill(kUnknown);

  // Index of `s`, interning it if new; kMemoFull if the cap is reached.
  const auto intern = [&](State&& s) -> std::int32_t {
    for (std::size_t i = 0; i < states.size(); ++i) {
      if (states_equal(states[i], s)) return static_cast<std::int32_t>(i);
    }
    if (states.size() >= kMaxMemoStates) return kMemoFull;
    states.push_back(std::move(s));
    trans.emplace_back();
    trans.back().fill(kUnknown);
    return static_cast<std::int32_t>(states.size() - 1);
  };

  std::vector<std::pair<std::uint32_t, std::int32_t>> work;
  work.emplace_back(0U, 0);
  while (!work.empty()) {
    const auto [node, sidx] = work.back();
    work.pop_back();
    ++last_nodes_visited_;
    const TokenTrie::Node& n = trie.nodes_[node];
    for (const std::int32_t id : n.tokens) mask[static_cast<std::size_t>(id)] = 1;
    for (const auto& [byte, child] : n.children) {
      std::int32_t t = trans[static_cast<std::size_t>(sidx)][byte];
      if (t == kUnknown) {
        State next;
        if (step(states[static_cast<std::size_t>(sidx)], byte, next)) {
          t = intern(std::move(next));
          if (t == kMemoFull) {  // too many distinct states: use the direct walk
            allowed_tokens_direct(trie, mask);
            return;
          }
        } else {
          t = kReject;
        }
        trans[static_cast<std::size_t>(sidx)][byte] = t;
      }
      if (t >= 0) work.emplace_back(child, t);
    }
  }
}

TokenTrie::TokenTrie(const std::vector<std::string>& token_bytes) : vocab_size_(token_bytes.size()) {
  nodes_.emplace_back();  // root: the empty prefix
  // Insert in byte order, so each node's children come out sorted and a
  // token only has to walk the part it doesn't share with the previous one.
  std::vector<std::size_t> order;
  order.reserve(token_bytes.size());
  for (std::size_t id = 0; id < token_bytes.size(); ++id) {
    if (!token_bytes[id].empty()) order.push_back(id);
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) { return token_bytes[a] < token_bytes[b]; });
  std::vector<std::uint32_t> path{0};  // path[k] = node for the first k bytes of `prev`
  std::string_view prev;
  for (const std::size_t id : order) {
    const std::string_view bytes = token_bytes[id];
    std::size_t common = 0;
    while (common < prev.size() && common < bytes.size() && prev[common] == bytes[common]) ++common;
    path.resize(common + 1);
    for (std::size_t k = common; k < bytes.size(); ++k) {
      const auto child = static_cast<std::uint32_t>(nodes_.size());
      nodes_.emplace_back();
      nodes_[path.back()].children.emplace_back(static_cast<std::uint8_t>(bytes[k]), child);
      path.push_back(child);
    }
    nodes_[path.back()].tokens.push_back(static_cast<std::int32_t>(id));
    prev = bytes;
  }
}

}  // namespace llmi::grammar
