#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llmi/grammar/grammar.hpp"

namespace llmi::grammar {

// Constrained decoding, part 2: which bytes a grammar allows next (M8,
// ADR 0009).
//
// The matcher runs the grammar as a pushdown automaton over Unicode code
// points, the same design as llama.cpp's grammar engine: its state is a set
// of stacks of grammar positions, one per way the input so far can still be
// parsed. Every stack is kept "normalized" -- its top is a character set
// waiting to be matched -- and an empty stack means the input so far is a
// complete sentence of the grammar.
//
// Model tokens are bytes, not characters, and one token can end in the
// middle of a multi-byte UTF-8 character. The matcher therefore takes bytes,
// holds an incomplete character back, and accepts a partial one only if some
// stack could still match a character that starts with those bytes.
//
// Ambiguous grammars can multiply stacks; the matcher fails closed when the
// count or a stack's depth passes its limits (the input is rejected, never
// accepted unchecked).
class TokenTrie;

struct MatcherLimits {
  std::size_t max_stacks = 2048;
  std::size_t max_stack_depth = 1024;
  std::size_t max_expansions = 100'000;  // rule expansions per character
};

class Matcher {
 public:
  using Limits = MatcherLimits;

  explicit Matcher(std::shared_ptr<const Grammar> grammar, Limits limits = {});

  // Feeds `bytes`. Returns false, leaving the state unchanged, if the
  // grammar does not allow them.
  bool accept(std::string_view bytes);
  [[nodiscard]] bool can_accept(std::string_view bytes) const;

  // The input so far is a complete sentence (and no character is half-read).
  [[nodiscard]] bool complete() const;
  // No continuation is possible at all.
  [[nodiscard]] bool dead() const { return state_.stacks.empty(); }
  // At least one more character is possible (whether or not the input so
  // far is already complete).
  [[nodiscard]] bool has_continuation() const;

  // mask[id] = 1 if token `id`'s bytes may come next, 0 if not, for every
  // token in `trie` (tokens with no bytes, such as special tokens, are 0;
  // whether to allow end-of-sequence is the caller's decision, from
  // complete()). `mask` is resized to trie.vocab_size().
  void allowed_tokens(const TokenTrie& trie, std::vector<std::uint8_t>& mask) const;

  // Statistics for the performance study: trie nodes visited by the last
  // allowed_tokens() call.
  [[nodiscard]] std::size_t last_nodes_visited() const { return last_nodes_visited_; }

  struct Pos {
    std::uint32_t rule = 0;
    std::uint32_t alt = 0;
    std::uint32_t idx = 0;
    friend bool operator==(const Pos&, const Pos&) = default;
    friend auto operator<=>(const Pos&, const Pos&) = default;
  };
  using Stack = std::vector<Pos>;

  struct State {
    std::vector<Stack> stacks;
    std::array<std::uint8_t, 4> partial{};  // an incomplete UTF-8 character
    std::uint8_t partial_len = 0;
  };

 private:
  [[nodiscard]] bool step(const State& in, std::uint8_t byte, State& out) const;
  [[nodiscard]] bool advance(const std::vector<Stack>& in, char32_t cp, std::vector<Stack>& out) const;
  [[nodiscard]] bool normalize(Stack start, std::vector<Stack>& out) const;
  [[nodiscard]] bool viable(const std::vector<Stack>& stacks, char32_t lo, char32_t hi) const;
  [[nodiscard]] const Symbol& symbol_at(const Pos& p) const;

  std::shared_ptr<const Grammar> grammar_;
  Limits limits_;
  State state_;
  mutable std::size_t last_nodes_visited_ = 0;
};

// Every token's bytes, arranged as a prefix tree so a grammar check can
// reject a whole subtree of tokens at the first byte it disallows.
class TokenTrie {
 public:
  // token_bytes[id] = the raw bytes token `id` decodes to; an empty string
  // means "never allowed by a grammar" (special tokens).
  explicit TokenTrie(const std::vector<std::string>& token_bytes);

  [[nodiscard]] std::size_t vocab_size() const { return vocab_size_; }
  [[nodiscard]] std::size_t node_count() const { return nodes_.size(); }

 private:
  struct Node {
    std::vector<std::pair<std::uint8_t, std::uint32_t>> children;  // sorted by byte
    std::vector<std::int32_t> tokens;  // tokens whose bytes end exactly here
  };
  std::vector<Node> nodes_;
  std::size_t vocab_size_ = 0;
  friend class Matcher;
};

}  // namespace llmi::grammar
