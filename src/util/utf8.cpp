#include "llmi/util/utf8.hpp"

namespace llmi::utf8 {

std::optional<char32_t> next(std::string_view s, std::size_t& i) {
  if (i >= s.size()) return std::nullopt;
  const auto b0 = static_cast<unsigned char>(s[i]);
  if (b0 < 0x80) {
    ++i;
    return b0;
  }
  std::size_t len = 0;
  char32_t cp = 0;
  char32_t min = 0;
  if ((b0 & 0xE0U) == 0xC0U) {
    len = 2;
    cp = b0 & 0x1FU;
    min = 0x80;
  } else if ((b0 & 0xF0U) == 0xE0U) {
    len = 3;
    cp = b0 & 0x0FU;
    min = 0x800;
  } else if ((b0 & 0xF8U) == 0xF0U) {
    len = 4;
    cp = b0 & 0x07U;
    min = 0x10000;
  } else {
    return std::nullopt;
  }
  if (s.size() - i < len) return std::nullopt;
  for (std::size_t k = 1; k < len; ++k) {
    const auto b = static_cast<unsigned char>(s[i + k]);
    if ((b & 0xC0U) != 0x80U) return std::nullopt;
    cp = (cp << 6U) | (b & 0x3FU);
  }
  if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return std::nullopt;
  i += len;
  return cp;
}

bool valid(std::string_view s) {
  std::size_t i = 0;
  while (i < s.size()) {
    if (!next(s, i)) return false;
  }
  return true;
}

std::optional<std::vector<char32_t>> decode(std::string_view s) {
  std::vector<char32_t> out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    auto cp = next(s, i);
    if (!cp) return std::nullopt;
    out.push_back(*cp);
  }
  return out;
}

void append(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
    out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
    out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
  } else {
    out.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
    out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
  }
}

}  // namespace llmi::utf8
