// bench_constrained: microbenchmarks the constrained-decoding mask -- the
// Matcher::allowed_tokens() trie walk that decides which of the ~49k tokens a
// grammar allows at each step (M8, ADR 0009, docs/constrained-decoding.md).
// It needs no model: it drives each grammar along a valid token path chosen
// from the real tokenizer's vocabulary, the same states the engine reaches
// when it actually generates, and times the mask at every step. This is the
// profiling step behind the M8 performance study (docs/evidence/
// m8-mask-profile.txt), the same idea as tools/bench_kernels.cpp for M4.
//
//   g++ -O3 -std=c++20 -I include \
//       tools/bench_constrained.cpp src/grammar/matcher.cpp \
//       src/grammar/grammar.cpp src/grammar/json_schema.cpp \
//       src/tokenizer/tokenizer.cpp src/util/json.cpp src/util/utf8.cpp \
//       -o bench_constrained && ./bench_constrained testdata/smollm2-135m/tokenizer.json
//
// Optional second argument: repetitions used to time each mask (default 25).

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "llmi/grammar/grammar.hpp"
#include "llmi/grammar/json_schema.hpp"
#include "llmi/grammar/matcher.hpp"
#include "llmi/tokenizer/tokenizer.hpp"
#include "llmi/util/json.hpp"
#include "llmi/util/result.hpp"

// tokenizer.cpp defines Tokenizer::load(), which references llmi::read_file
// (defined in model.cpp). This benchmark loads the tokenizer with from_json()
// instead and does not want model.cpp's dependencies, so it provides its own
// read_file definition to satisfy the link.
namespace llmi {
Result<std::string> read_file(const std::string& path, std::size_t max_bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return fail("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  std::string s = ss.str();
  if (s.size() > max_bytes) return fail("file too large: " + path);
  return s;
}
}  // namespace llmi

namespace {

// Heap-allocation counter. Toggled on only for the untimed allocation pass, so
// it never perturbs a timing measurement.
struct AllocCounters {
  bool on = false;
  std::size_t count = 0;
  std::size_t bytes = 0;
};
AllocCounters g_alloc;

using Clock = std::chrono::steady_clock;

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double median_sizes(std::vector<std::size_t> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return static_cast<double>(v[v.size() / 2]);
}

// Every token's raw bytes, exactly as grammar::Vocabulary builds them: special
// tokens get no bytes, so a grammar never allows them.
std::vector<std::string> token_bytes_of(const llmi::Tokenizer& tok) {
  std::vector<std::string> out(tok.vocab_size());
  for (std::size_t id = 0; id < out.size(); ++id) {
    const auto t = static_cast<llmi::TokenId>(id);
    if (tok.is_special(t)) continue;
    if (auto b = tok.decode({t}, false)) out[id] = std::move(b.value());
  }
  return out;
}

std::size_t count_allowed(const std::vector<std::uint8_t>& mask) {
  std::size_t n = 0;
  for (const auto m : mask) n += (m != 0);
  return n;
}

struct GrammarCase {
  std::string name;
  std::string gbnf;
};

// The five JSON schemas from tools/eval_constrained_json.py, so the mask cost
// measured here lines up with the end-to-end validity numbers, plus the
// any-JSON-object grammar and one hand-written non-JSON GBNF.
llmi::grammar::Grammar parse_or_die(const std::string& name, const std::string& gbnf) {
  auto g = llmi::grammar::Grammar::parse(gbnf);
  if (!g) {
    std::fprintf(stderr, "grammar %s failed to parse: %s\n", name.c_str(), g.error().c_str());
    std::exit(1);
  }
  return std::move(g.value());
}

std::string schema_to_gbnf_or_die(const std::string& name, const std::string& schema_json) {
  auto v = llmi::json::parse(schema_json);
  if (!v) {
    std::fprintf(stderr, "schema %s is not JSON: %s\n", name.c_str(), v.error().c_str());
    std::exit(1);
  }
  auto gbnf = llmi::grammar::json_schema_to_gbnf(*v);
  if (!gbnf) {
    std::fprintf(stderr, "schema %s did not convert: %s\n", name.c_str(), gbnf.error().c_str());
    std::exit(1);
  }
  return std::move(gbnf.value());
}

// One mask measurement: drive the grammar along a valid token path (a fixed
// RNG picks among the allowed tokens each step, so the walk is representative
// and reproducible), timing allowed_tokens() at every step. Returns per-step
// vectors.
struct Walk {
  std::vector<double> ms;       // median time of each mask
  std::vector<std::size_t> nodes;
  std::vector<std::size_t> allowed;
  std::size_t steps = 0;
  bool completed = false;
};

Walk walk_grammar(const llmi::grammar::Grammar& grammar, const llmi::grammar::TokenTrie& trie,
                  const std::vector<std::string>& token_bytes, int reps, std::size_t max_steps,
                  std::uint64_t seed) {
  Walk w;
  llmi::grammar::Matcher m(std::make_shared<const llmi::grammar::Grammar>(grammar));
  std::mt19937_64 rng(seed);
  std::vector<std::uint8_t> mask;
  for (std::size_t step = 0; step < max_steps; ++step) {
    // Time the mask: the deterministic trie walk, repeated to get a stable
    // median. Nodes visited and allowed count are deterministic.
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    for (int r = 0; r < reps; ++r) {
      const auto t0 = Clock::now();
      m.allowed_tokens(trie, mask);
      samples.push_back(std::chrono::duration<double>(Clock::now() - t0).count() * 1000.0);
    }
    const std::size_t allowed = count_allowed(mask);
    w.ms.push_back(median(samples));
    w.nodes.push_back(m.last_nodes_visited());
    w.allowed.push_back(allowed);

    // Collect allowed token ids and choose one to advance.
    std::vector<llmi::TokenId> ids;
    ids.reserve(allowed);
    for (std::size_t id = 0; id < mask.size(); ++id) {
      if (mask[id] != 0 && !token_bytes[id].empty()) ids.push_back(static_cast<llmi::TokenId>(id));
    }
    if (ids.empty()) break;  // dead end
    // Once a complete sentence is reachable and we have generated a few
    // tokens, stop with modest probability so walks end naturally rather than
    // running to the step cap on unbounded grammars.
    if (m.complete() && step >= 8 && (rng() % 4U) == 0U) {
      w.completed = true;
      break;
    }
    const llmi::TokenId pick = ids[rng() % ids.size()];
    if (!m.accept(token_bytes[static_cast<std::size_t>(pick)])) break;
    w.steps = step + 1;
    if (m.complete() && !m.has_continuation()) {
      w.completed = true;
      break;
    }
  }
  return w;
}

// Untimed pass over the same walk, counting heap allocations, to see whether
// the mask's cost is dominated by per-node State allocation.
void alloc_pass(const llmi::grammar::Grammar& grammar, const llmi::grammar::TokenTrie& trie,
                const std::vector<std::string>& token_bytes, std::size_t max_steps, std::uint64_t seed,
                std::size_t& out_allocs, std::size_t& out_bytes, std::size_t& out_nodes) {
  llmi::grammar::Matcher m(std::make_shared<const llmi::grammar::Grammar>(grammar));
  std::mt19937_64 rng(seed);
  std::vector<std::uint8_t> mask;
  std::size_t nodes = 0;
  g_alloc = {};
  g_alloc.on = true;
  for (std::size_t step = 0; step < max_steps; ++step) {
    m.allowed_tokens(trie, mask);
    nodes += m.last_nodes_visited();
    std::vector<llmi::TokenId> ids;
    for (std::size_t id = 0; id < mask.size(); ++id) {
      if (mask[id] != 0 && !token_bytes[id].empty()) ids.push_back(static_cast<llmi::TokenId>(id));
    }
    if (ids.empty()) break;
    if (m.complete() && step >= 8 && (rng() % 4U) == 0U) break;
    const llmi::TokenId pick = ids[rng() % ids.size()];
    if (!m.accept(token_bytes[static_cast<std::size_t>(pick)])) break;
    if (m.complete() && !m.has_continuation()) break;
  }
  g_alloc.on = false;
  out_allocs = g_alloc.count;
  out_bytes = g_alloc.bytes;
  out_nodes = nodes;
}

}  // namespace

void* operator new(std::size_t n) {
  if (g_alloc.on) {
    ++g_alloc.count;
    g_alloc.bytes += n;
  }
  void* p = std::malloc(n == 0 ? 1 : n);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main(int argc, char** argv) {
  const std::string tok_path = argc > 1 ? argv[1] : "testdata/smollm2-135m/tokenizer.json";
  const int reps = argc > 2 ? std::atoi(argv[2]) : 25;

  std::ifstream in(tok_path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open tokenizer %s\n", tok_path.c_str());
    return 1;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string tok_json = ss.str();
  auto tok = llmi::Tokenizer::from_json(tok_json);
  if (!tok) {
    std::fprintf(stderr, "cannot load tokenizer %s: %s\n", tok_path.c_str(), tok.error().c_str());
    return 1;
  }
  const std::vector<std::string> token_bytes = token_bytes_of(tok.value());

  const auto t_trie = Clock::now();
  const llmi::grammar::TokenTrie trie(token_bytes);
  const double trie_ms = std::chrono::duration<double>(Clock::now() - t_trie).count() * 1000.0;

  std::size_t with_bytes = 0;
  for (const auto& b : token_bytes) with_bytes += !b.empty();

  std::printf("tokenizer: %s\n", tok_path.c_str());
  std::printf("vocabulary: %zu tokens (%zu with bytes a grammar can allow), trie %zu nodes built in %.1f ms\n",
              token_bytes.size(), with_bytes, trie.node_count(), trie_ms);
  std::printf("timing: median of %d repetitions per mask; one mask per generated token\n\n", reps);

  auto person = R"({"type":"object","properties":{"name":{"type":"string","maxLength":40},"age":{"type":"integer"},"city":{"type":"string","maxLength":40}},"required":["name","age","city"],"additionalProperties":false})";
  auto maintenance = R"({"type":"object","properties":{"vehicle":{"type":"string","maxLength":30},"mileage":{"type":"integer"},"status":{"enum":["ok","service_due","out_of_service"]},"parts":{"type":"array","items":{"type":"string","maxLength":30},"maxItems":3}},"required":["vehicle","mileage","status","parts"],"additionalProperties":false})";
  auto shopping = R"({"type":"object","properties":{"items":{"type":"array","minItems":1,"maxItems":4,"items":{"type":"object","properties":{"name":{"type":"string","maxLength":20},"qty":{"type":"integer"}},"required":["name","qty"],"additionalProperties":false}}},"required":["items"],"additionalProperties":false})";

  std::vector<GrammarCase> cases;
  cases.push_back({"person (schema)", schema_to_gbnf_or_die("person", person)});
  cases.push_back({"maintenance (schema, enum+array)", schema_to_gbnf_or_die("maintenance", maintenance)});
  cases.push_back({"shopping-list (schema, nested)", schema_to_gbnf_or_die("shopping", shopping)});
  cases.push_back({"any JSON object", llmi::grammar::json_object_grammar()});
  cases.push_back({"word list (GBNF)",
                   "root ::= \"[\" ws item (ws \",\" ws item)* ws \"]\"\n"
                   "item ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
                   "ws ::= [ \\t\\n]*\n"});

  // "Heavy" = a mask that walks more than half the trie (the free-string case
  // where almost every token is allowed). These few masks dominate the total.
  const std::size_t heavy_nodes = trie.node_count() / 2;
  std::printf("%-34s %6s %9s %9s %9s %7s %9s %9s\n", "grammar", "masks", "median ms", "mean ms", "max ms", "heavy",
              "med node", "max node");
  std::printf("%-34s %6s %9s %9s %9s %7s %9s %9s\n", "-------", "-----", "--------", "-------", "------", "-----",
              "--------", "--------");
  for (const auto& c : cases) {
    const llmi::grammar::Grammar g = parse_or_die(c.name, c.gbnf);
    const Walk w = walk_grammar(g, trie, token_bytes, reps, 160, 20250101ULL);
    double total = 0.0;
    std::size_t heavy = 0;
    for (std::size_t i = 0; i < w.ms.size(); ++i) {
      total += w.ms[i];
      if (w.nodes[i] >= heavy_nodes) ++heavy;
    }
    const double mean = w.ms.empty() ? 0.0 : total / static_cast<double>(w.ms.size());
    std::printf("%-34s %6zu %9.3f %9.3f %9.3f %7zu %9zu %9zu\n", c.name.c_str(), w.ms.size(), median(w.ms), mean,
                (w.ms.empty() ? 0.0 : *std::max_element(w.ms.begin(), w.ms.end())), heavy,
                static_cast<std::size_t>(median_sizes(w.nodes)),
                (w.nodes.empty() ? 0UL : *std::max_element(w.nodes.begin(), w.nodes.end())));
  }
  std::printf("(\"heavy\" = masks that walked >= %zu trie nodes, i.e. most of the vocabulary)\n", heavy_nodes);

  std::printf("\nallocation profile (one full walk per grammar, heap allocations counted):\n");
  std::printf("%-34s %10s %12s %12s %10s\n", "grammar", "nodes", "allocations", "alloc/node", "bytes/node");
  std::printf("%-34s %10s %12s %12s %10s\n", "-------", "-----", "-----------", "----------", "----------");
  for (const auto& c : cases) {
    const llmi::grammar::Grammar g = parse_or_die(c.name, c.gbnf);
    std::size_t allocs = 0;
    std::size_t bytes = 0;
    std::size_t nodes = 0;
    alloc_pass(g, trie, token_bytes, 160, 20250101ULL, allocs, bytes, nodes);
    std::printf("%-34s %10zu %12zu %12.2f %10.1f\n", c.name.c_str(), nodes, allocs,
                nodes ? static_cast<double>(allocs) / static_cast<double>(nodes) : 0.0,
                nodes ? static_cast<double>(bytes) / static_cast<double>(nodes) : 0.0);
  }
  return 0;
}
