#include "llmi/grammar/constraint.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

#include "llmi/model/transformer.hpp"

namespace llmi::grammar {

namespace {

std::vector<std::string> all_token_bytes(const Tokenizer& tokenizer) {
  std::vector<std::string> out(tokenizer.vocab_size());
  for (std::size_t id = 0; id < out.size(); ++id) {
    const auto t = static_cast<TokenId>(id);
    if (tokenizer.is_special(t)) continue;
    if (auto bytes = tokenizer.decode({t}, false)) out[id] = std::move(bytes.value());
  }
  return out;
}

}  // namespace

Vocabulary::Vocabulary(const Tokenizer& tokenizer) : token_bytes(all_token_bytes(tokenizer)), trie(token_bytes) {}

Constraint::Constraint(std::shared_ptr<const Grammar> grammar, std::shared_ptr<const Vocabulary> vocab,
                       std::vector<TokenId> eos_ids)
    : vocab_(std::move(vocab)), matcher_(std::move(grammar)), eos_ids_(std::move(eos_ids)) {}

bool Constraint::is_eos(TokenId id) const { return std::find(eos_ids_.begin(), eos_ids_.end(), id) != eos_ids_.end(); }

bool Constraint::allowed(TokenId id) const {
  if (is_eos(id)) return matcher_.complete();
  if (id < 0 || static_cast<std::size_t>(id) >= vocab_->token_bytes.size()) return false;
  const std::string& bytes = vocab_->token_bytes[static_cast<std::size_t>(id)];
  return !bytes.empty() && matcher_.can_accept(bytes);
}

Result<TokenId> Constraint::pick(std::vector<float>& logits, const SamplingConfig& cfg, std::mt19937_64& rng) {
  using Clock = std::chrono::steady_clock;
  ++stats_.picks;
  const auto t0 = Clock::now();
  if (cfg.temperature <= 0.0F) {
    const auto best = static_cast<TokenId>(argmax(logits.data(), logits.size()));
    if (allowed(best)) {
      stats_.mask_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
      return best;
    }
  }
  matcher_.allowed_tokens(vocab_->trie, mask_);
  ++stats_.full_masks;
  stats_.trie_nodes_visited += matcher_.last_nodes_visited();
  // An end-of-sequence id means only that, even in a vocabulary where the
  // same id also has bytes of its own: allowed exactly when the output is
  // complete, never as text.
  const std::uint8_t eos_allowed = matcher_.complete() ? 1 : 0;
  for (const TokenId e : eos_ids_) {
    if (e >= 0 && static_cast<std::size_t>(e) < mask_.size()) mask_[static_cast<std::size_t>(e)] = eos_allowed;
  }
  stats_.mask_seconds += std::chrono::duration<double>(Clock::now() - t0).count();

  bool any = false;
  const std::size_t n = std::min(logits.size(), mask_.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (i < n && mask_[i] != 0) {
      any = true;
    } else {
      logits[i] = -std::numeric_limits<float>::infinity();
    }
  }
  if (!any) return fail("no token in this vocabulary can continue the grammar");
  return sample(logits.data(), logits.size(), cfg, rng);
}

bool Constraint::accept(TokenId id) {
  if (is_eos(id)) return matcher_.complete();
  if (id < 0 || static_cast<std::size_t>(id) >= vocab_->token_bytes.size()) return false;
  const std::string& bytes = vocab_->token_bytes[static_cast<std::size_t>(id)];
  return !bytes.empty() && matcher_.accept(bytes);
}

bool Constraint::finished() const { return matcher_.complete() && !matcher_.has_continuation(); }

}  // namespace llmi::grammar
