#include "llmi/model/transformer.hpp"

#include <algorithm>
#include <cmath>

#include "llmi/model/kernels.hpp"

namespace llmi {

namespace {

std::vector<float> to_f32(const SafetensorsFile& w, const std::string& name) {
  const TensorInfo* t = w.find(name);  // presence and shape were checked by Model
  const auto raw = w.data(*t);
  std::vector<float> out(static_cast<std::size_t>(t->numel()));
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = read_float(raw, t->dtype, i);
  return out;
}

}  // namespace

std::size_t argmax(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);  // first maximum
}

Result<Transformer> Transformer::load(const Model& model) {
  Transformer t;
  t.cfg_ = model.config();
  const auto& c = t.cfg_;
  const auto& w = model.weights();
  t.embed_ = to_f32(w, "model.embed_tokens.weight");
  t.final_norm_ = to_f32(w, "model.norm.weight");
  if (!c.tie_word_embeddings) t.lm_head_ = to_f32(w, "lm_head.weight");
  t.layers_.resize(c.num_layers);
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    Layer& l = t.layers_[i];
    l.attn_norm = to_f32(w, p + "input_layernorm.weight");
    l.wq = to_f32(w, p + "self_attn.q_proj.weight");
    l.wk = to_f32(w, p + "self_attn.k_proj.weight");
    l.wv = to_f32(w, p + "self_attn.v_proj.weight");
    if (c.qkv_bias) {
      l.bq = to_f32(w, p + "self_attn.q_proj.bias");
      l.bk = to_f32(w, p + "self_attn.k_proj.bias");
      l.bv = to_f32(w, p + "self_attn.v_proj.bias");
    }
    l.wo = to_f32(w, p + "self_attn.o_proj.weight");
    l.mlp_norm = to_f32(w, p + "post_attention_layernorm.weight");
    l.w_gate = to_f32(w, p + "mlp.gate_proj.weight");
    l.w_up = to_f32(w, p + "mlp.up_proj.weight");
    l.w_down = to_f32(w, p + "mlp.down_proj.weight");
  }
  // inv_freq[i] = 1 / theta^(2i/d), computed in float32 like the reference.
  const std::size_t d = c.head_dim;
  t.inv_freq_.resize(d / 2);
  for (std::size_t i = 0; i < d / 2; ++i) {
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(d);
    t.inv_freq_[i] = 1.0F / std::pow(static_cast<float>(c.rope_theta), exponent);
  }
  return t;
}

Result<std::vector<float>> Transformer::forward(const std::vector<TokenId>& tokens, bool all_positions,
                                                ForwardTrace* trace) const {
  const auto& c = cfg_;
  const std::size_t T = tokens.size();
  const std::size_t H = c.hidden_size;
  const std::size_t hd = c.head_dim;
  const std::size_t nh = c.num_heads;
  const std::size_t nkv = c.num_kv_heads;
  const std::size_t group = nh / nkv;
  const std::size_t F = c.intermediate_size;
  const std::size_t V = c.vocab_size;
  const auto eps = static_cast<float>(c.rms_norm_eps);
  if (T == 0) return fail("forward: no tokens");
  if (group == 0 || nh % nkv != 0) return fail("forward: inconsistent head counts");  // parse_config rules this out
  if (T > c.max_position_embeddings) return fail("forward: sequence longer than the model's context");
  for (const TokenId id : tokens) {
    if (id < 0 || static_cast<std::size_t>(id) >= V) return fail("forward: token id " + std::to_string(id) + " out of range");
  }

  std::vector<float> x(T * H);
  for (std::size_t t = 0; t < T; ++t) {
    std::copy_n(embed_.data() + static_cast<std::size_t>(tokens[t]) * H, H, x.data() + t * H);
  }
  if (trace != nullptr) {
    trace->states.clear();
    trace->states.push_back(x);
  }

  std::vector<float> h(T * H);
  std::vector<float> q(T * nh * hd);
  std::vector<float> k(T * nkv * hd);
  std::vector<float> v(T * nkv * hd);
  std::vector<float> attn(T * nh * hd);
  std::vector<float> proj(T * H);
  std::vector<float> gate(T * F);
  std::vector<float> up(T * F);
  std::vector<float> scores(T);
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));

  for (const Layer& l : layers_) {
    // Attention block.
    for (std::size_t t = 0; t < T; ++t) kernels::rmsnorm(x.data() + t * H, l.attn_norm.data(), H, eps, h.data() + t * H);
    kernels::matmul(h.data(), T, H, l.wq.data(), nh * hd, l.bq.empty() ? nullptr : l.bq.data(), q.data());
    kernels::matmul(h.data(), T, H, l.wk.data(), nkv * hd, l.bk.empty() ? nullptr : l.bk.data(), k.data());
    kernels::matmul(h.data(), T, H, l.wv.data(), nkv * hd, l.bv.empty() ? nullptr : l.bv.data(), v.data());
    for (std::size_t t = 0; t < T; ++t) {
      for (std::size_t hh = 0; hh < nh; ++hh) kernels::rope(q.data() + (t * nh + hh) * hd, hd, t, inv_freq_.data());
      for (std::size_t hh = 0; hh < nkv; ++hh) kernels::rope(k.data() + (t * nkv + hh) * hd, hd, t, inv_freq_.data());
    }
    for (std::size_t hh = 0; hh < nh; ++hh) {
      const std::size_t kvh = hh / group;  // grouped-query attention: heads share key/value heads
      for (std::size_t t = 0; t < T; ++t) {
        const float* qv = q.data() + (t * nh + hh) * hd;
        for (std::size_t s = 0; s <= t; ++s) {  // causal: position t sees 0..t
          scores[s] = kernels::dot(qv, k.data() + (s * nkv + kvh) * hd, hd) * scale;
        }
        kernels::softmax(scores.data(), t + 1);
        float* out = attn.data() + (t * nh + hh) * hd;
        std::fill_n(out, hd, 0.0F);
        for (std::size_t s = 0; s <= t; ++s) {
          const float p = scores[s];
          const float* vv = v.data() + (s * nkv + kvh) * hd;
          for (std::size_t i = 0; i < hd; ++i) out[i] += p * vv[i];
        }
      }
    }
    kernels::matmul(attn.data(), T, nh * hd, l.wo.data(), H, nullptr, proj.data());
    for (std::size_t i = 0; i < T * H; ++i) x[i] += proj[i];

    // MLP block.
    for (std::size_t t = 0; t < T; ++t) kernels::rmsnorm(x.data() + t * H, l.mlp_norm.data(), H, eps, h.data() + t * H);
    kernels::matmul(h.data(), T, H, l.w_gate.data(), F, nullptr, gate.data());
    kernels::matmul(h.data(), T, H, l.w_up.data(), F, nullptr, up.data());
    for (std::size_t i = 0; i < T * F; ++i) gate[i] = kernels::silu(gate[i]) * up[i];
    kernels::matmul(gate.data(), T, F, l.w_down.data(), H, nullptr, proj.data());
    for (std::size_t i = 0; i < T * H; ++i) x[i] += proj[i];

    if (trace != nullptr) trace->states.push_back(x);
  }

  for (std::size_t t = 0; t < T; ++t) kernels::rmsnorm(x.data() + t * H, final_norm_.data(), H, eps, h.data() + t * H);
  if (trace != nullptr) trace->states.push_back(h);

  const float* head = lm_head_.empty() ? embed_.data() : lm_head_.data();
  const std::size_t first = all_positions ? 0 : T - 1;
  std::vector<float> logits((T - first) * V);
  kernels::matmul(h.data() + first * H, T - first, H, head, V, nullptr, logits.data());
  return logits;
}

Result<std::vector<TokenId>> Transformer::generate_greedy(std::vector<TokenId> tokens, std::size_t max_new,
                                                          const std::vector<TokenId>& stop_ids) const {
  std::vector<TokenId> generated;
  for (std::size_t step = 0; step < max_new; ++step) {
    auto logits = forward(tokens);
    if (!logits) return fail(logits.error());
    const auto next = static_cast<TokenId>(argmax(logits->data(), logits->size()));
    generated.push_back(next);
    tokens.push_back(next);
    if (std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end()) break;
  }
  return generated;
}

Result<KVCache> Transformer::new_cache(std::size_t max_len) const {
  KVCache cache;
  cache.capacity_ = std::min(max_len, static_cast<std::size_t>(cfg_.max_position_embeddings));
  if (cache.capacity_ == 0) return fail("new_cache: max_len must be positive");
  const std::size_t per_pos = static_cast<std::size_t>(cfg_.num_kv_heads) * cfg_.head_dim;
  cache.k_.assign(cfg_.num_layers, std::vector<float>(cache.capacity_ * per_pos));
  cache.v_.assign(cfg_.num_layers, std::vector<float>(cache.capacity_ * per_pos));
  return cache;
}

Result<std::vector<float>> Transformer::forward_cached(const std::vector<TokenId>& tokens, KVCache& cache,
                                                       bool all_positions, ForwardTrace* trace) const {
  const auto& c = cfg_;
  const std::size_t T = tokens.size();
  const std::size_t H = c.hidden_size;
  const std::size_t hd = c.head_dim;
  const std::size_t nh = c.num_heads;
  const std::size_t nkv = c.num_kv_heads;
  const std::size_t group = nh / nkv;
  const std::size_t F = c.intermediate_size;
  const std::size_t V = c.vocab_size;
  const auto eps = static_cast<float>(c.rms_norm_eps);
  if (T == 0) return fail("forward_cached: no tokens");
  if (group == 0 || nh % nkv != 0) return fail("forward_cached: inconsistent head counts");
  if (cache.k_.size() != c.num_layers) return fail("forward_cached: cache was not created by this model");
  const std::size_t base = cache.length_;  // position of the first new token
  if (base + T > cache.capacity_) return fail("forward_cached: cache is full");
  if (base + T > c.max_position_embeddings) return fail("forward_cached: sequence longer than the model's context");
  for (const TokenId id : tokens) {
    if (id < 0 || static_cast<std::size_t>(id) >= V) return fail("forward_cached: token id " + std::to_string(id) + " out of range");
  }

  std::vector<float> x(T * H);
  for (std::size_t t = 0; t < T; ++t) {
    std::copy_n(embed_.data() + static_cast<std::size_t>(tokens[t]) * H, H, x.data() + t * H);
  }
  if (trace != nullptr) {
    trace->states.clear();
    trace->states.push_back(x);
  }

  std::vector<float> h(T * H);
  std::vector<float> q(T * nh * hd);
  std::vector<float> attn(T * nh * hd);
  std::vector<float> proj(T * H);
  std::vector<float> gate(T * F);
  std::vector<float> up(T * F);
  std::vector<float> scores(base + T);
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));

  for (std::size_t li = 0; li < layers_.size(); ++li) {
    const Layer& l = layers_[li];
    float* k_cache = cache.k_[li].data();
    float* v_cache = cache.v_[li].data();

    // Attention block: compute Q for the new positions and K/V for the new
    // positions only, appending K/V into the cache; attend over 0..base+t.
    for (std::size_t t = 0; t < T; ++t) kernels::rmsnorm(x.data() + t * H, l.attn_norm.data(), H, eps, h.data() + t * H);
    kernels::matmul(h.data(), T, H, l.wq.data(), nh * hd, l.bq.empty() ? nullptr : l.bq.data(), q.data());
    kernels::matmul(h.data(), T, H, l.wk.data(), nkv * hd, l.bk.empty() ? nullptr : l.bk.data(),
                    k_cache + base * nkv * hd);
    kernels::matmul(h.data(), T, H, l.wv.data(), nkv * hd, l.bv.empty() ? nullptr : l.bv.data(),
                    v_cache + base * nkv * hd);
    for (std::size_t t = 0; t < T; ++t) {
      for (std::size_t hh = 0; hh < nh; ++hh) kernels::rope(q.data() + (t * nh + hh) * hd, hd, base + t, inv_freq_.data());
      for (std::size_t hh = 0; hh < nkv; ++hh) kernels::rope(k_cache + ((base + t) * nkv + hh) * hd, hd, base + t, inv_freq_.data());
    }
    for (std::size_t hh = 0; hh < nh; ++hh) {
      const std::size_t kvh = hh / group;  // grouped-query attention: heads share key/value heads
      for (std::size_t t = 0; t < T; ++t) {
        const std::size_t pos = base + t;  // global position, causal: sees 0..pos
        const float* qv = q.data() + (t * nh + hh) * hd;
        for (std::size_t s = 0; s <= pos; ++s) {
          scores[s] = kernels::dot(qv, k_cache + (s * nkv + kvh) * hd, hd) * scale;
        }
        kernels::softmax(scores.data(), pos + 1);
        float* out = attn.data() + (t * nh + hh) * hd;
        std::fill_n(out, hd, 0.0F);
        for (std::size_t s = 0; s <= pos; ++s) {
          const float p = scores[s];
          const float* vv = v_cache + (s * nkv + kvh) * hd;
          for (std::size_t i = 0; i < hd; ++i) out[i] += p * vv[i];
        }
      }
    }
    kernels::matmul(attn.data(), T, nh * hd, l.wo.data(), H, nullptr, proj.data());
    for (std::size_t i = 0; i < T * H; ++i) x[i] += proj[i];

    // MLP block.
    for (std::size_t t = 0; t < T; ++t) kernels::rmsnorm(x.data() + t * H, l.mlp_norm.data(), H, eps, h.data() + t * H);
    kernels::matmul(h.data(), T, H, l.w_gate.data(), F, nullptr, gate.data());
    kernels::matmul(h.data(), T, H, l.w_up.data(), F, nullptr, up.data());
    for (std::size_t i = 0; i < T * F; ++i) gate[i] = kernels::silu(gate[i]) * up[i];
    kernels::matmul(gate.data(), T, F, l.w_down.data(), H, nullptr, proj.data());
    for (std::size_t i = 0; i < T * H; ++i) x[i] += proj[i];

    if (trace != nullptr) trace->states.push_back(x);
  }
  cache.length_ = base + T;

  for (std::size_t t = 0; t < T; ++t) kernels::rmsnorm(x.data() + t * H, final_norm_.data(), H, eps, h.data() + t * H);
  if (trace != nullptr) trace->states.push_back(h);

  const float* head = lm_head_.empty() ? embed_.data() : lm_head_.data();
  const std::size_t first = all_positions ? 0 : T - 1;
  std::vector<float> logits((T - first) * V);
  kernels::matmul(h.data() + first * H, T - first, H, head, V, nullptr, logits.data());
  return logits;
}

Result<std::vector<TokenId>> Transformer::generate_greedy_cached(const std::vector<TokenId>& prompt,
                                                                 std::size_t max_new,
                                                                 const std::vector<TokenId>& stop_ids) const {
  auto cache = new_cache(prompt.size() + max_new);
  if (!cache) return fail(cache.error());
  auto logits = forward_cached(prompt, *cache);
  if (!logits) return fail(logits.error());

  std::vector<TokenId> generated;
  auto next = static_cast<TokenId>(argmax(logits->data(), logits->size()));
  for (std::size_t step = 0; step < max_new; ++step) {
    generated.push_back(next);
    if (std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end()) break;
    if (step + 1 == max_new) break;
    auto step_logits = forward_cached({next}, *cache);
    if (!step_logits) return fail(step_logits.error());
    next = static_cast<TokenId>(argmax(step_logits->data(), step_logits->size()));
  }
  return generated;
}

}  // namespace llmi
