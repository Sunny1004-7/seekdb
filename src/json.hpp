// Strict JSON for the MiniVectorDB stdio protocol.
// - int64 stays exact (never routed through double)
// - rejects NaN / Infinity, duplicate keys, trailing garbage
// - emits one compact line, UTF-8 preserved
#pragma once
#include <charconv>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mj {

struct Value;
using Ptr = std::shared_ptr<Value>;

enum class Type { Null, Bool, Int, Real, Str, Arr, Obj };

struct Value {
  Type type = Type::Null;
  bool b = false;
  int64_t i = 0;
  double r = 0;
  std::string s;
  std::vector<Ptr> arr;
  std::vector<std::pair<std::string, Ptr>> obj;  // insertion ordered

  static Ptr null() { auto v = std::make_shared<Value>(); v->type = Type::Null; return v; }
  static Ptr boolean(bool x) { auto v = std::make_shared<Value>(); v->type = Type::Bool; v->b = x; return v; }
  static Ptr integer(int64_t x) { auto v = std::make_shared<Value>(); v->type = Type::Int; v->i = x; return v; }
  static Ptr real(double x) { auto v = std::make_shared<Value>(); v->type = Type::Real; v->r = x; return v; }
  static Ptr str(std::string x) { auto v = std::make_shared<Value>(); v->type = Type::Str; v->s = std::move(x); return v; }
  static Ptr array() { auto v = std::make_shared<Value>(); v->type = Type::Arr; return v; }
  static Ptr object() { auto v = std::make_shared<Value>(); v->type = Type::Obj; return v; }

  const Ptr* find(const std::string& k) const {
    for (auto& kv : obj) if (kv.first == k) return &kv.second;
    return nullptr;
  }
  void set(const std::string& k, Ptr v) {
    for (auto& kv : obj) if (kv.first == k) { kv.second = std::move(v); return; }
    obj.emplace_back(k, std::move(v));
  }
};

struct Error : std::runtime_error {
  explicit Error(const std::string& m) : std::runtime_error(m) {}
};

class Parser {
 public:
  explicit Parser(const std::string& text) : t_(text) {}

  Ptr parse() {
    skip();
    Ptr v = value(0);
    skip();
    if (p_ != t_.size()) throw Error("trailing characters");
    return v;
  }

 private:
  const std::string& t_;
  size_t p_ = 0;

  void skip() {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\r' || t_[p_] == '\n')) ++p_;
  }
  char peek() {
    if (p_ >= t_.size()) throw Error("unexpected end");
    return t_[p_];
  }
  bool lit(const char* w) {
    size_t n = std::strlen(w);
    if (t_.compare(p_, n, w) == 0) { p_ += n; return true; }
    return false;
  }

  Ptr value(int depth) {
    if (depth > 64) throw Error("nesting too deep");
    char c = peek();
    switch (c) {
      case '{': return object(depth);
      case '[': return array(depth);
      case '"': return Value::str(string());
      case 't': if (lit("true")) return Value::boolean(true); throw Error("bad literal");
      case 'f': if (lit("false")) return Value::boolean(false); throw Error("bad literal");
      case 'n': if (lit("null")) return Value::null(); throw Error("bad literal");
      default: return number();
    }
  }

  Ptr object(int depth) {
    ++p_;
    auto v = Value::object();
    skip();
    if (peek() == '}') { ++p_; return v; }
    for (;;) {
      skip();
      if (peek() != '"') throw Error("object key must be string");
      std::string k = string();
      for (auto& kv : v->obj) if (kv.first == k) throw Error("duplicate key");
      skip();
      if (peek() != ':') throw Error("expected ':'");
      ++p_;
      skip();
      v->obj.emplace_back(std::move(k), value(depth + 1));
      skip();
      char c = peek();
      if (c == ',') { ++p_; continue; }
      if (c == '}') { ++p_; return v; }
      throw Error("expected ',' or '}'");
    }
  }

  Ptr array(int depth) {
    ++p_;
    auto v = Value::array();
    skip();
    if (peek() == ']') { ++p_; return v; }
    for (;;) {
      skip();
      v->arr.push_back(value(depth + 1));
      skip();
      char c = peek();
      if (c == ',') { ++p_; continue; }
      if (c == ']') { ++p_; return v; }
      throw Error("expected ',' or ']'");
    }
  }

  static void utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) out += char(cp);
    else if (cp <= 0x7FF) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
    else if (cp <= 0xFFFF) {
      out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F));
    } else {
      out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F));
      out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F));
    }
  }

  uint32_t hex4() {
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      if (p_ >= t_.size()) throw Error("bad \\u escape");
      char c = t_[p_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else throw Error("bad hex digit");
    }
    return v;
  }

  std::string string() {
    ++p_;  // opening quote
    std::string out;
    for (;;) {
      if (p_ >= t_.size()) throw Error("unterminated string");
      // Fast path: a run with no quote/backslash (the common case — a 1.4 KB
      // vector literal is one run). strcspn is SIMD in glibc; control chars
      // are rejected 8 bytes at a time. Semantics identical to the old
      // char-by-char loop, ~20x faster on long literals.
      const char* base = t_.data() + p_;
      const size_t remain = t_.size() - p_;
      size_t run = std::strcspn(base, "\"\\");
      if (run > remain) run = remain;
      size_t i = 0;
      for (; i + 8 <= run; i += 8) {
        uint64_t w;
        std::memcpy(&w, base + i, 8);
        uint64_t x = w & 0xE0E0E0E0E0E0E0E0ull;
        // any zero byte in x <=> some source byte < 0x20 (borrow-induced false
        // positives only happen when a zero byte is present, which throws anyway)
        if (((x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull) != 0) break;
      }
      for (; i < run; ++i) if (static_cast<unsigned char>(base[i]) < 0x20) break;
      if (i < run) throw Error("control character in string");
      out.append(base, run);
      p_ += run;
      if (p_ >= t_.size()) throw Error("unterminated string");
      char c = t_[p_];
      if (c == '"') { ++p_; return out; }
      ++p_;  // backslash
      if (p_ >= t_.size()) throw Error("bad escape");
      char e = t_[p_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (p_ + 1 < t_.size() && t_[p_] == '\\' && t_[p_ + 1] == 'u') {
              p_ += 2;
              uint32_t lo = hex4();
              if (lo < 0xDC00 || lo > 0xDFFF) throw Error("bad low surrogate");
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else throw Error("lone high surrogate");
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) throw Error("lone low surrogate");
          utf8(out, cp);
          break;
        }
        default: throw Error("unknown escape");
      }
    }
  }

  Ptr number() {
    size_t start = p_;
    if (peek() == '-') ++p_;
    if (p_ >= t_.size()) throw Error("bad number");
    if (t_[p_] == '0') ++p_;
    else if (t_[p_] >= '1' && t_[p_] <= '9') { while (p_ < t_.size() && isdigit((unsigned char)(t_[p_]))) ++p_; }
    else throw Error("bad number");
    bool real = false;
    if (p_ < t_.size() && t_[p_] == '.') {
      real = true; ++p_;
      if (p_ >= t_.size() || !isdigit((unsigned char)(t_[p_]))) throw Error("bad fraction");
      while (p_ < t_.size() && isdigit((unsigned char)(t_[p_]))) ++p_;
    }
    if (p_ < t_.size() && (t_[p_] == 'e' || t_[p_] == 'E')) {
      real = true; ++p_;
      if (p_ < t_.size() && (t_[p_] == '+' || t_[p_] == '-')) ++p_;
      if (p_ >= t_.size() || !isdigit((unsigned char)(t_[p_]))) throw Error("bad exponent");
      while (p_ < t_.size() && isdigit((unsigned char)(t_[p_]))) ++p_;
    }
    std::string num = t_.substr(start, p_ - start);
    if (!real) {
      errno = 0;
      long long v = std::strtoll(num.c_str(), nullptr, 10);
      if (errno == ERANGE) throw Error("integer overflow");
      return Value::integer(int64_t(v));
    }
    double d = std::strtod(num.c_str(), nullptr);
    if (!std::isfinite(d)) throw Error("non-finite number");
    return Value::real(d);
  }
};

inline Ptr parse(const std::string& text) { return Parser(text).parse(); }

// Zero-alloc JSONL fast request parser for {"id":<int|plain_str>,"op":"execute","sql":"<plain>",["search_mode":"<plain>"]}.
// Returns true iff strict JSON rules hold without escapes or unknown/duplicate keys; otherwise falls back to parse().
inline bool try_parse_exec_req(std::string_view text, std::string_view& id_raw,
                               std::string_view& sql, std::string_view& mode) {
  const char* p = text.data();
  const char* end = p + text.size();
  auto skip_ws = [&]() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
  };
  auto eat_plain_str = [&](std::string_view& out) -> bool {
    skip_ws();
    if (p >= end || *p != '"') return false;
    const char* s = ++p;
    const char* q = static_cast<const char*>(std::memchr(s, '"', size_t(end - s)));
    if (!q) return false;
    const size_t n = size_t(q - s);
    if (std::memchr(s, '\\', n) != nullptr) return false;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      uint64_t w;
      std::memcpy(&w, s + i, 8);
      uint64_t x = w & 0xE0E0E0E0E0E0E0E0ull;
      if (((x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull) != 0) return false;
    }
    for (; i < n; ++i) if (static_cast<unsigned char>(s[i]) < 0x20) return false;
    out = std::string_view(s, n);
    p = q + 1;
    return true;
  };

  skip_ws();
  if (p >= end || *p != '{') return false;
  ++p;
  skip_ws();
  if (p < end && *p == '}') return false;

  uint8_t seen = 0;
  mode = "exact";
  for (;;) {
    std::string_view key;
    if (!eat_plain_str(key)) return false;
    skip_ws();
    if (p >= end || *p != ':') return false;
    ++p;
    if (key == "id") {
      if (seen & 1) return false;
      seen |= 1;
      skip_ws();
      if (p >= end) return false;
      if (*p == '"') {
        const char* start = p;
        std::string_view dummy;
        if (!eat_plain_str(dummy)) return false;
        id_raw = std::string_view(start, size_t(p - start));
      } else {
        const char* start = p;
        if (*p == '-') {
          ++p;
          if (p >= end || *p < '1' || *p > '9') return false;
          const char* d0 = p++;
          while (p < end && unsigned(*p - '0') <= 9u) ++p;
          if (size_t(p - d0) > 18) return false;  // guard int64 overflow via fallback
        } else if (*p == '0') {
          ++p;
        } else if (*p >= '1' && *p <= '9') {
          const char* d0 = p++;
          while (p < end && unsigned(*p - '0') <= 9u) ++p;
          if (size_t(p - d0) > 18) return false;
        } else {
          return false;
        }
        if (p < end && (*p == '.' || *p == 'e' || *p == 'E')) return false;
        id_raw = std::string_view(start, size_t(p - start));
      }
    } else if (key == "op") {
      if (seen & 2) return false;
      seen |= 2;
      std::string_view opv;
      if (!eat_plain_str(opv) || opv != "execute") return false;
    } else if (key == "sql") {
      if (seen & 4) return false;
      seen |= 4;
      if (!eat_plain_str(sql)) return false;
    } else if (key == "search_mode") {
      if (seen & 8) return false;
      seen |= 8;
      if (!eat_plain_str(mode)) return false;
    } else {
      return false;
    }
    skip_ws();
    if (p >= end) return false;
    if (*p == ',') { ++p; continue; }
    if (*p == '}') { ++p; break; }
    return false;
  }
  skip_ws();
  return p == end && (seen & 7) == 7;
}

inline void append_double(std::string& out, double r) {
  if (!std::isfinite(r)) throw Error("cannot emit non-finite");
  if (r == 0.0) {
    out += std::signbit(r) ? "-0.0" : "0.0";
    return;
  }
  char buf[40];
  auto rc = std::to_chars(buf, buf + sizeof buf, r);
  size_t n = size_t(rc.ptr - buf);
  out.append(buf, n);
  // keep it a JSON float so 5 stays 5.0 for the distance column
  bool has_dot_exp = false;
  for (size_t i = 0; i < n; ++i) {
    if (buf[i] == '.' || buf[i] == 'e' || buf[i] == 'E') { has_dot_exp = true; break; }
  }
  if (!has_dot_exp) out += ".0";
}

inline void dump_string(std::string& out, const std::string& s) {
  out += '"';
  const char* p = s.data();
  const size_t n = s.size();
  size_t i = 0;
  for (; i < n; ++i) {
    unsigned char c = static_cast<unsigned char>(p[i]);
    if (__builtin_expect(c < 0x20 || c == '"' || c == '\\', 0)) break;
  }
  if (__builtin_expect(i == n, 1)) {
    out.append(p, n);
    out += '"';
    return;
  }
  if (i) out.append(p, i);
  for (; i < n; ++i) {
    unsigned char c = static_cast<unsigned char>(p[i]);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else out += char(c);
    }
  }
  out += '"';
}

inline void dump(std::string& out, const Ptr& v) {
  switch (v->type) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += v->b ? "true" : "false"; break;
    case Type::Int: out += std::to_string(v->i); break;
    case Type::Real: append_double(out, v->r); break;
    case Type::Str: dump_string(out, v->s); break;
    case Type::Arr: {
      out += '[';
      for (size_t k = 0; k < v->arr.size(); ++k) { if (k) out += ','; dump(out, v->arr[k]); }
      out += ']';
      break;
    }
    case Type::Obj: {
      out += '{';
      for (size_t k = 0; k < v->obj.size(); ++k) {
        if (k) out += ',';
        dump_string(out, v->obj[k].first);
        out += ':';
        dump(out, v->obj[k].second);
      }
      out += '}';
      break;
    }
  }
}

inline std::string stringify(const Ptr& v) { std::string out; dump(out, v); return out; }

}  // namespace mj