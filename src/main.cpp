// MiniVectorDB entry point: JSONL over stdin/stdout per STDIO_PROTOCOL.md.
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>
#include <utility>

#include "db.hpp"
#include "json.hpp"

static void write_all(const char* p, size_t n) {
  while (n > 0) {
    ssize_t w = ::write(STDOUT_FILENO, p, n);
    if (w > 0) {
      p += w;
      n -= static_cast<size_t>(w);
    } else if (w < 0 && errno == EINTR) {
      continue;
    } else {
      break;
    }
  }
}

static void emit(const mj::Ptr& v) {
  std::string line = mj::stringify(v);
  line += '\n';
  write_all(line.data(), line.size());
}

struct LineReader {
  std::string buf;
  size_t pos = 0;
  size_t scanned = 0;
  bool eof = false;

  bool next(std::string& out) {
    out.clear();
    while (true) {
      if (scanned < buf.size()) {
        const void* nl = std::memchr(buf.data() + scanned, '\n', buf.size() - scanned);
        if (nl) {
          size_t nl_idx = static_cast<const char*>(nl) - buf.data();
          size_t end = nl_idx;
          if (end > pos && buf[end - 1] == '\r') --end;
          out.assign(buf.data() + pos, end - pos);
          pos = nl_idx + 1;
          scanned = pos;
          if (pos == buf.size()) {
            if (buf.capacity() > (512u << 10)) std::string().swap(buf);
            else buf.clear();
            pos = 0;
            scanned = 0;
          } else if (pos > (128u << 10)) {
            buf.erase(0, pos);
            pos = 0;
            scanned = 0;
          }
          return true;
        }
        scanned = buf.size();
      }
      if (eof) {
        if (pos < buf.size()) {
          size_t end = buf.size();
          if (end > pos && buf[end - 1] == '\r') --end;
          out.assign(buf.data() + pos, end - pos);
          if (buf.capacity() > (512u << 10)) std::string().swap(buf);
          else buf.clear();
          pos = 0;
          scanned = 0;
          return true;
        }
        return false;
      }
      if (pos > 0 && buf.size() > (128u << 10) && pos >= buf.size() / 2) {
        buf.erase(0, pos);
        pos = 0;
        scanned = buf.size();
      }
      size_t old_sz = buf.size();
      constexpr size_t CHUNK = 65536;
      buf.resize(old_sz + CHUNK);
      ssize_t n;
      do {
        n = ::read(STDIN_FILENO, buf.data() + old_sz, CHUNK);
      } while (n < 0 && errno == EINTR);
      if (n <= 0) {
        buf.resize(old_sz);
        eof = true;
      } else {
        buf.resize(old_sz + static_cast<size_t>(n));
      }
    }
  }
};

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "/data";
  std::unique_ptr<db::Engine> engine;
  try {
    engine = std::make_unique<db::Engine>(dir);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "startup failed: %s\n", e.what());
    return 1;
  }

  {
    auto ready = mj::Value::object();
    ready->set("type", mj::Value::str("ready"));
    emit(ready);
  }

  LineReader reader;
  std::string line;
  std::string out_line;
  while (reader.next(line)) {
    if (line.empty()) continue;
    std::string_view id_raw, sql_sv, mode_sv;
    if (mj::try_parse_exec_req(line, id_raw, sql_sv, mode_sv)) {
      try {
        engine->execute_jsonl_sv(id_raw, sql_sv, mode_sv, out_line);
        write_all(out_line.data(), out_line.size());
        if (line.capacity() > (512u << 10)) std::string().swap(line);
        if (out_line.capacity() > (512u << 10)) std::string().swap(out_line);
      } catch (const std::exception& e) {
        out_line.clear();
        out_line += "{\"id\":";
        out_line.append(id_raw.data(), id_raw.size());
        out_line += ",\"ok\":false,\"error\":{\"code\":\"internal_error\",\"message\":";
        mj::dump_string(out_line, e.what());
        out_line += "}}\n";
        write_all(out_line.data(), out_line.size());
        if (line.capacity() > (512u << 10)) std::string().swap(line);
        if (out_line.capacity() > (512u << 10)) std::string().swap(out_line);
      }
      continue;
    }

    mj::Ptr req;
    try {
      req = mj::parse(line);
    } catch (const std::exception& e) {
      if (line.capacity() > (512u << 10)) std::string().swap(line);
      auto r = mj::Value::object();
      r->set("ok", mj::Value::boolean(false));
      auto err = mj::Value::object();
      err->set("code", mj::Value::str("protocol_error"));
      err->set("message", mj::Value::str(e.what()));
      r->set("error", err);
      emit(r);
      continue;
    }
    if (line.capacity() > (512u << 10)) std::string().swap(line);
    const mj::Ptr* idp = req->find("id");
    mj::Ptr id = idp ? *idp : mj::Value::null();
    const mj::Ptr* opp = req->find("op");
    std::string op = opp && (*opp)->type == mj::Type::Str ? (*opp)->s : "";

    if (op == "shutdown") {
      try {
        engine->shutdown();
      } catch (const std::exception& e) {
        std::fprintf(stderr, "shutdown: %s\n", e.what());
      }
      auto r = mj::Value::object();
      r->set("id", id);
      r->set("ok", mj::Value::boolean(true));
      emit(r);
      return 0;
    }

    if (op != "execute") {
      auto r = mj::Value::object();
      r->set("id", id);
      r->set("ok", mj::Value::boolean(false));
      auto err = mj::Value::object();
      err->set("code", mj::Value::str("protocol_error"));
      err->set("message", mj::Value::str("unknown op"));
      r->set("error", err);
      emit(r);
      continue;
    }

    const mj::Ptr* sqlp = req->find("sql");
    const mj::Ptr* modep = req->find("search_mode");
    std::string text = sqlp && (*sqlp)->type == mj::Type::Str ? std::move((*sqlp)->s) : "";
    std::string mode = modep && (*modep)->type == mj::Type::Str ? std::move((*modep)->s) : "exact";
    req.reset();

    try {
      engine->execute_jsonl(id, text, mode, out_line);
      write_all(out_line.data(), out_line.size());
      if (out_line.capacity() > (512u << 10)) std::string().swap(out_line);
    } catch (const std::exception& e) {
      auto r = mj::Value::object();
      r->set("id", id);
      r->set("ok", mj::Value::boolean(false));
      auto err = mj::Value::object();
      err->set("code", mj::Value::str("internal_error"));
      err->set("message", mj::Value::str(e.what()));
      r->set("error", err);
      emit(r);
    }
  }

  engine->shutdown();
  return 0;
}