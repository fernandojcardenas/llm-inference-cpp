#pragma once

#include <memory>
#include <string>
#include <vector>

#include "llmi/util/result.hpp"

namespace llmi::chat {

struct Message {
  std::string role;     // "system", "user" or "assistant"
  std::string content;
};

// A minimal Jinja2 interpreter covering exactly the constructs Hugging
// Face's chat templates use: {{ expr }} output, {% for x in y %}/{% endfor
// %}, {% if %}/{% elif %}/{% else %}/{% endif %}, {% set %}, {%- -%}
// whitespace control, attribute (a.b) and index (a['b'], a[0]) access,
// string literals with \n-style escapes, "+" concatenation, "==", "!=",
// "and", "or", "not", and "x is [not] defined". It is not a general Jinja
// engine (no macros, no tojson rendering beyond a stub, no arithmetic
// besides what parses) -- see docs/chat-template.md for exactly what's
// covered and why that's enough for the two models this engine runs.
//
// This engine does not call tools, so anything a template does only when
// tools or a message's tool_calls are present never executes: those values
// are always undefined here, exactly as when a reference caller invokes
// apply_chat_template() without passing tools.
class ChatTemplate {
 public:
  ChatTemplate() = default;  // default-constructed: render() fails until assigned from parse()/load()

  static Result<ChatTemplate> parse(const std::string& jinja_source);

  // Reads a tokenizer_config.json and parses its "chat_template" field.
  static Result<ChatTemplate> load(const std::string& tokenizer_config_path);

  [[nodiscard]] Result<std::string> render(const std::vector<Message>& messages, bool add_generation_prompt) const;

  struct Node;  // Defined in template.cpp; public only so the parser there can name it.

 private:
  std::shared_ptr<const std::vector<Node>> root_;
};

}  // namespace llmi::chat
