#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace llmi::utf8 {

// Decodes one code point starting at s[i]. Strict: rejects overlong forms,
// surrogates, values above U+10FFFF and truncated sequences. On success,
// returns the code point and advances i.
std::optional<char32_t> next(std::string_view s, std::size_t& i);

// True if the whole string is valid UTF-8.
bool valid(std::string_view s);

// Decodes a valid UTF-8 string into code points; nullopt if invalid.
std::optional<std::vector<char32_t>> decode(std::string_view s);

void append(std::string& out, char32_t cp);

}  // namespace llmi::utf8
