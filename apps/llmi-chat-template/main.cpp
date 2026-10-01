// llmi-chat-template: renders a chat prompt from a tokenizer_config.json's
// chat_template and a JSON message list, for tools/crosscheck_chat_template.py
// and for apps/llmi-chat to build its prompt.
//
//   llmi-chat-template TOKENIZER_CONFIG_JSON MESSAGES_JSON [--no-generation-prompt]
//
// MESSAGES_JSON is a JSON array of {"role": ..., "content": ...} objects.
// Prints the rendered prompt to stdout with no trailing newline added.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "llmi/chat/template.hpp"
#include "llmi/util/json.hpp"

namespace {

bool read_file(const std::string& path, std::string* out) {
  const std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s TOKENIZER_CONFIG_JSON MESSAGES_JSON [--no-generation-prompt]\n", argv[0]);
    return 2;
  }
  bool add_generation_prompt = true;
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "--no-generation-prompt") add_generation_prompt = false;
  }

  auto tpl = llmi::chat::ChatTemplate::load(argv[1]);
  if (!tpl) {
    std::fprintf(stderr, "error: %s\n", tpl.error().c_str());
    return 1;
  }

  std::string messages_text;
  if (!read_file(argv[2], &messages_text)) {
    std::fprintf(stderr, "error: cannot read %s\n", argv[2]);
    return 1;
  }
  auto doc = llmi::json::parse(messages_text);
  if (!doc || !doc->is_array()) {
    std::fprintf(stderr, "error: %s must be a JSON array of messages\n", argv[2]);
    return 1;
  }
  std::vector<llmi::chat::Message> messages;
  for (const auto& item : doc->items()) {
    const auto* role = item.find("role");
    const auto* content = item.find("content");
    if (role == nullptr || content == nullptr || !role->is_string() || !content->is_string()) {
      std::fprintf(stderr, "error: each message needs string \"role\" and \"content\"\n");
      return 1;
    }
    messages.push_back({role->text(), content->text()});
  }

  auto out = tpl->render(messages, add_generation_prompt);
  if (!out) {
    std::fprintf(stderr, "error: %s\n", out.error().c_str());
    return 1;
  }
  std::fwrite(out->data(), 1, out->size(), stdout);
  return 0;
}
