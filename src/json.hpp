// Strict JSON for the MiniVectorDB stdio protocol.
// - int64 stays exact (never routed through double)
// - rejects NaN / Infinity, duplicate keys, trailing garbage
// - emits one compact line, UTF-8 preserved
#pragma once
#include <cstdint>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
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
      unsigned char c = t_[p_++];
      if (c == '"') return out;
      if (c == '\\') {
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
      } else {
        if (c < 0x20) throw Error("control character in string");
        out += char(c);
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

inline void dump_string(std::string& out, const std::string& s) {
  out += '"';
  for (unsigned char c : s) {
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
    case Type::Real: {
      if (!std::isfinite(v->r)) throw Error("cannot emit non-finite");
      char buf[40];
      std::snprintf(buf, sizeof buf, "%.17g", v->r);
      out += buf;
      // keep it a JSON float so 5 stays 5.0 for the distance column
      if (!std::strpbrk(buf, ".eE")) out += ".0";
      break;
    }
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