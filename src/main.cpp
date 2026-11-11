// MiniVectorDB entry point: JSONL over stdin/stdout per STDIO_PROTOCOL.md.
#include <cstdio>
#include <iostream>
#include <string>

#include "db.hpp"
#include "json.hpp"

static void emit(const mj::Ptr& v) {
  std::string line = mj::stringify(v);
  line += '\n';
  ::fwrite(line.data(), 1, line.size(), stdout);
  ::fflush(stdout);
}

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

  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    mj::Ptr req;
    try {
      req = mj::parse(line);
    } catch (const std::exception& e) {
      auto r = mj::Value::object();
      r->set("ok", mj::Value::boolean(false));
      auto err = mj::Value::object();
      err->set("code", mj::Value::str("protocol_error"));
      err->set("message", mj::Value::str(e.what()));
      r->set("error", err);
      emit(r);
      continue;
    }
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
    std::string text = sqlp && (*sqlp)->type == mj::Type::Str ? (*sqlp)->s : "";
    std::string mode = modep && (*modep)->type == mj::Type::Str ? (*modep)->s : "exact";

    mj::Ptr body;
    try {
      body = engine->execute(text, mode);
    } catch (const std::exception& e) {
      body = mj::Value::object();
      body->set("ok", mj::Value::boolean(false));
      auto err = mj::Value::object();
      err->set("code", mj::Value::str("internal_error"));
      err->set("message", mj::Value::str(e.what()));
      body->set("error", err);
    }
    auto r = mj::Value::object();
    r->set("id", id);
    for (const auto& kv : body->obj) r->set(kv.first, kv.second);
    emit(r);
  }

  engine->shutdown();
  return 0;
}