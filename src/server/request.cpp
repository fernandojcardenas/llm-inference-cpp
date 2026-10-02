#include "llmi/server/request.hpp"

#include "llmi/util/json.hpp"

namespace llmi::server {

namespace {

bool is_valid_role(const std::string& role) { return role == "system" || role == "user" || role == "assistant"; }

// Reads an optional numeric field, applying `check` to the double value.
// Returns false (via *out_error) when the field is present but not a
// number, or fails `check`; leaves `out` untouched when the field is
// absent, so the caller's default stands.
template <typename Check>
bool read_optional_number(const json::Value& obj, std::string_view key, float& out, Check check,
                           std::string* out_error) {
  const json::Value* v = obj.find(key);
  if (v == nullptr) return true;
  const auto d = v->as_double();
  if (!d || !check(*d)) {
    *out_error = std::string(key) + " is invalid";
    return false;
  }
  out = static_cast<float>(*d);
  return true;
}

}  // namespace

Result<ChatCompletionRequest> parse_chat_completion_request(std::string_view body, const RequestLimits& limits) {
  auto parsed = json::parse(body);
  if (!parsed) return fail("invalid JSON request body: " + parsed.error());
  const json::Value& root = *parsed;
  if (!root.is_object()) return fail("request body must be a JSON object");

  ChatCompletionRequest req;
  req.max_tokens = limits.default_max_tokens;

  if (const json::Value* model = root.find("model"); model != nullptr) {
    if (!model->is_string()) return fail("\"model\" must be a string");
    req.model = model->text();
  }

  const json::Value* messages = root.find("messages");
  if (messages == nullptr || !messages->is_array()) return fail("\"messages\" must be a non-empty array");
  if (messages->items().empty()) return fail("\"messages\" must not be empty");
  if (messages->items().size() > limits.max_messages) return fail("too many messages");
  req.messages.reserve(messages->items().size());
  for (const json::Value& m : messages->items()) {
    if (!m.is_object()) return fail("each message must be a JSON object");
    const json::Value* role = m.find("role");
    const json::Value* content = m.find("content");
    if (role == nullptr || !role->is_string() || !is_valid_role(role->text())) {
      return fail(R"(each message's "role" must be "system", "user" or "assistant")");
    }
    if (content == nullptr || !content->is_string()) return fail(R"(each message's "content" must be a string)");
    if (content->text().size() > limits.max_message_bytes) return fail("a message's content is too large");
    req.messages.push_back(chat::Message{role->text(), content->text()});
  }

  if (const json::Value* stream = root.find("stream"); stream != nullptr) {
    if (!stream->is_bool()) return fail(R"("stream" must be a boolean)");
    req.stream = stream->as_bool();
  }

  if (const json::Value* max_tokens = root.find("max_tokens"); max_tokens != nullptr) {
    const auto n = max_tokens->as_u64();
    if (!n) return fail(R"("max_tokens" must be a non-negative integer)");
    if (*n > limits.max_tokens_ceiling) return fail("\"max_tokens\" exceeds this server's limit");
    req.max_tokens = static_cast<std::size_t>(*n);
  }

  std::string error;
  if (!read_optional_number(root, "temperature", req.sampling.temperature,
                             [](double d) { return d >= 0.0 && d <= 2.0; }, &error)) {
    return fail(std::move(error));
  }
  if (!read_optional_number(root, "top_p", req.sampling.top_p, [](double d) { return d >= 0.0 && d <= 1.0; },
                             &error)) {
    return fail(std::move(error));
  }
  if (const json::Value* top_k = root.find("top_k"); top_k != nullptr) {
    const auto n = top_k->as_u64();
    if (!n) return fail(R"("top_k" must be a non-negative integer)");
    req.sampling.top_k = static_cast<std::size_t>(*n);
  }

  if (const json::Value* stop = root.find("stop"); stop != nullptr) {
    if (stop->is_string()) {
      req.stop.push_back(stop->text());
    } else if (stop->is_array()) {
      if (stop->items().size() > limits.max_stop_sequences) return fail("too many \"stop\" sequences");
      for (const json::Value& s : stop->items()) {
        if (!s.is_string()) return fail(R"("stop" entries must be strings)");
        req.stop.push_back(s.text());
      }
    } else {
      return fail(R"("stop" must be a string or an array of strings)");
    }
  }

  return req;
}

}  // namespace llmi::server
