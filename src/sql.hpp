// SQL subset for MiniVectorDB: lexer + recursive-descent parser.
// Anything outside the agreed subset raises unsupported_sql (never silently
// reinterpreted), malformed text raises syntax_error.
#pragma once
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sql {

struct Error : std::runtime_error {
  std::string code;
  Error(std::string c, const std::string& m) : std::runtime_error(m), code(std::move(c)) {}
};
[[noreturn]] inline void unsupported(const std::string& m) { throw Error("unsupported_sql", m); }
[[noreturn]] inline void syntax(const std::string& m) { throw Error("syntax_error", m); }

enum class ColType { BigInt, VarChar, Vector };

struct ColumnDef {
  std::string name;
  ColType type = ColType::BigInt;
  int len = 0;  // VARCHAR(n) chars, VECTOR(n) dims
  bool primary = false;
};

enum class ExprKind { IntLit, StrLit, Column, L2, Cmp, And };
enum class CmpOp { Eq, Lt, Le, Gt, Ge };

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct Expr {
  ExprKind kind = ExprKind::IntLit;
  int64_t ival = 0;
  std::string sval;             // string literal / column name
  CmpOp cmp = CmpOp::Eq;
  ExprPtr lhs, rhs;
};

struct SelectItem {
  ExprPtr expr;
  std::string alias;  // empty => derive from expression
  bool star = false;
};

struct OrderKey {
  std::string name;  // column name or select alias
  bool desc = false;
};

enum class StmtKind { CreateTable, CreateVectorIndex, Insert, Select, Update, Delete, Begin, Commit, Rollback };

struct Statement {
  StmtKind kind = StmtKind::Select;
  std::string table;
  std::string index_name;
  std::string index_column;
  std::vector<ColumnDef> columns;                 // CREATE TABLE
  std::vector<std::string> insert_columns;        // INSERT (optional list)
  std::vector<std::vector<ExprPtr>> rows;         // INSERT VALUES
  std::vector<SelectItem> items;                  // SELECT
  ExprPtr where;                                  // SELECT / UPDATE / DELETE
  std::vector<OrderKey> order;
  int64_t limit = -1;
  bool has_limit = false;
  std::vector<std::pair<std::string, ExprPtr>> sets;  // UPDATE
};

// ---------------------------------------------------------------- lexer

enum class Tok { End, Ident, Number, String, Punct };

// Zero-copy tokens: `text` is a view into the statement being parsed (the
// parser never outlives it). Only identifiers pay for an uppercased copy, and
// only string literals containing escapes pay for a decoded copy — a 100-row
// INSERT used to allocate two strings per numeric token, now it allocates none.
struct Token {
  Tok kind = Tok::End;
  std::string_view text;  // raw slice; String: inner content, escapes intact
  std::string upper;      // Ident only, for keyword matching
  std::string decoded;    // String only, when has_decoded
  bool has_decoded = false;
};

inline std::string upper(std::string s) {
  for (char& c : s) if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
  return s;
}

class Lexer {
 public:
  explicit Lexer(const std::string& s) : s_(s) {}

  std::vector<Token> run() {
    std::vector<Token> out;
    // one allocation: ~1 token per 12 source chars covers VALUES-heavy INSERTs
    out.reserve(s_.size() / 12 + 8);
    for (;;) {
      while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\r' || s_[p_] == '\n')) ++p_;
      if (p_ >= s_.size()) break;
      char c = s_[p_];
      if (c == '-' && p_ + 1 < s_.size() && s_[p_ + 1] == '-') {  // line comment
        while (p_ < s_.size() && s_[p_] != '\n') ++p_;
        continue;
      }
      Token t;
      if (c == '\'') {
        size_t st = ++p_;
        bool esc = false;
        for (;;) {
          if (p_ >= s_.size()) syntax("unterminated string literal");
          char d = s_[p_++];
          if (d == '\'') {
            if (p_ < s_.size() && s_[p_] == '\'') { esc = true; ++p_; continue; }
            break;
          }
          if (d == '\\') {  // tolerate backslash escapes
            esc = true;
            if (p_ < s_.size()) ++p_;
          }
        }
        t.kind = Tok::String;
        t.text = std::string_view(s_).substr(st, p_ - 1 - st);
        if (esc) {
          t.decoded = decode(std::string(t.text));
          t.has_decoded = true;
        }
      } else if ((c >= '0' && c <= '9') ||
                 (c == '-' && p_ + 1 < s_.size() && s_[p_ + 1] >= '0' && s_[p_ + 1] <= '9') ||
                 (c == '.' && p_ + 1 < s_.size() && s_[p_ + 1] >= '0' && s_[p_ + 1] <= '9')) {
        size_t st = p_;
        if (c == '-') ++p_;
        // Same accepted character set as before, but digit runs (the bulk of
        // a 100-row INSERT) advance one compare per char instead of six.
        const size_t n = s_.size();
        for (;;) {
          while (p_ < n && (unsigned char)(s_[p_] - '0') <= 9) ++p_;
          if (p_ >= n) break;
          char d = s_[p_];
          if (d == '.' || d == 'e' || d == 'E') { ++p_; continue; }
          if ((d == '+' || d == '-') && p_ > st && (s_[p_ - 1] == 'e' || s_[p_ - 1] == 'E')) { ++p_; continue; }
          break;
        }
        t.kind = Tok::Number;
        t.text = std::string_view(s_).substr(st, p_ - st);
      } else if (isalpha((unsigned char)(c)) || c == '_' || (unsigned char)c >= 0x80) {
        size_t st = p_;
        while (p_ < s_.size() && (isalnum((unsigned char)(s_[p_])) || s_[p_] == '_' || s_[p_] == '$' ||
                                  (unsigned char)s_[p_] >= 0x80)) ++p_;
        t.kind = Tok::Ident;
        t.text = std::string_view(s_).substr(st, p_ - st);
        t.upper = upper(std::string(t.text));
      } else {
        // multi-char comparison operators
        if ((c == '<' || c == '>' || c == '!' ) && p_ + 1 < s_.size() && s_[p_ + 1] == '=') {
          t.kind = Tok::Punct;
          t.text = std::string_view(s_).substr(p_, 2);
          p_ += 2;
        } else if (c == '<' && p_ + 1 < s_.size() && s_[p_ + 1] == '>') {
          t.kind = Tok::Punct; t.text = std::string_view(s_).substr(p_, 2); p_ += 2;
        } else {
          t.kind = Tok::Punct;
          t.text = std::string_view(s_).substr(p_, 1);
          ++p_;
        }
      }
      out.push_back(std::move(t));
    }
    out.push_back(Token{});
    return out;
  }

 private:
  const std::string& s_;
  size_t p_ = 0;

  // Legacy escape semantics, applied only to literals that contain escapes.
  static std::string decode(const std::string& v) {
    std::string out;
    out.reserve(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
      char d = v[i];
      if (d == '\'' && i + 1 < v.size() && v[i + 1] == '\'') { out += '\''; ++i; continue; }
      if (d == '\\' && i + 1 < v.size()) {
        char e = v[++i];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case '0': out += '\0'; break;
          default: out += e;
        }
        continue;
      }
      out += d;
    }
    return out;
  }
};

// ---------------------------------------------------------------- parser

class Parser {
 public:
  explicit Parser(const std::string& text) { toks_ = Lexer(text).run(); }

  Statement parse() {
    Statement st;
    const std::string kw = cur().upper;
    if (kw == "CREATE") {
      ++i_;
      if (cur().upper == "TABLE") { ++i_; st.kind = StmtKind::CreateTable; create_table(st); }
      else if (cur().upper == "VECTOR") {
        ++i_;
        expect_kw("INDEX");
        st.kind = StmtKind::CreateVectorIndex;
        st.index_name = ident();
        expect_kw("ON");
        st.table = ident();
        expect("(");
        st.index_column = ident();
        expect(")");
      } else if (cur().upper == "INDEX" || cur().upper == "UNIQUE") unsupported("only VECTOR INDEX is supported");
      else unsupported("unsupported CREATE form");
    } else if (kw == "INSERT") {
      ++i_; st.kind = StmtKind::Insert; insert(st);
    } else if (kw == "SELECT") {
      ++i_; st.kind = StmtKind::Select; select(st);
    } else if (kw == "UPDATE") {
      ++i_; st.kind = StmtKind::Update; update(st);
    } else if (kw == "DELETE") {
      ++i_; st.kind = StmtKind::Delete;
      expect_kw("FROM");
      st.table = ident();
      if (cur().upper == "WHERE") { ++i_; st.where = or_expr(); }
    } else if (kw == "BEGIN") {
      ++i_; st.kind = StmtKind::Begin;
      if (cur().upper == "WORK" || cur().upper == "TRANSACTION") ++i_;
    } else if (kw == "START") {
      ++i_; expect_kw("TRANSACTION"); st.kind = StmtKind::Begin;
    } else if (kw == "COMMIT") {
      ++i_; st.kind = StmtKind::Commit;
      if (cur().upper == "WORK" || cur().upper == "TRANSACTION") ++i_;
    } else if (kw == "ROLLBACK") {
      ++i_; st.kind = StmtKind::Rollback;
      if (cur().upper == "WORK" || cur().upper == "TRANSACTION") ++i_;
    } else if (kw.empty()) {
      syntax("empty statement");
    } else {
      unsupported("statement not supported: " + kw);
    }
    if (cur().text == ";") ++i_;
    if (cur().kind != Tok::End) syntax("unexpected trailing input");
    return st;
  }

 private:
  std::vector<Token> toks_;
  size_t i_ = 0;

  const Token& cur() const { return toks_[i_]; }
  void expect(const std::string& p) {
    if (cur().text != p) syntax("expected '" + p + "'");
    ++i_;
  }
  void expect_kw(const std::string& k) {
    if (cur().upper != k) syntax("expected " + k);
    ++i_;
  }
  std::string ident() {
    if (cur().kind != Tok::Ident) syntax("expected identifier");
    std::string s(cur().text);
    ++i_;
    if (cur().text == ".") {  // qualified name: keep the last part
      ++i_;
      if (cur().kind != Tok::Ident) syntax("expected identifier after '.'");
      s = cur().text;
      ++i_;
    }
    return s;
  }

  void create_table(Statement& st) {
    st.table = ident();
    expect("(");
    bool pk_seen = false;
    int vec_seen = 0;
    for (;;) {
      if (cur().upper == "PRIMARY") {
        ++i_; expect_kw("KEY"); expect("(");
        std::string k = ident();
        expect(")");
        bool found = false;
        for (auto& c : st.columns) if (c.name == k) { c.primary = true; found = true; }
        if (!found) syntax("primary key on unknown column");
        pk_seen = true;
      } else {
        ColumnDef cd;
        cd.name = ident();
        std::string ty = cur().upper;
        ++i_;
        if (ty == "BIGINT" || ty == "INT" || ty == "INTEGER") {
          if (ty != "BIGINT") unsupported("only BIGINT integer columns are supported");
          cd.type = ColType::BigInt;
        } else if (ty == "VARCHAR" || ty == "CHAR") {
          if (ty != "VARCHAR") unsupported("only VARCHAR string columns are supported");
          cd.type = ColType::VarChar;
          expect("(");
          cd.len = int(number_int());
          expect(")");
          if (cd.len <= 0) syntax("VARCHAR length must be positive");
        } else if (ty == "VECTOR") {
          cd.type = ColType::Vector;
          expect("(");
          cd.len = int(number_int());
          expect(")");
          if (cd.len <= 0) syntax("VECTOR dimension must be positive");
          ++vec_seen;
        } else {
          unsupported("column type not supported: " + ty);
        }
        for (;;) {
          if (cur().upper == "PRIMARY") { ++i_; expect_kw("KEY"); cd.primary = true; pk_seen = true; continue; }
          if (cur().upper == "NOT") { ++i_; expect_kw("NULL"); continue; }
          if (cur().upper == "NULL") unsupported("NULL is not supported");
          if (cur().upper == "UNIQUE" || cur().upper == "DEFAULT" || cur().upper == "AUTO_INCREMENT")
            unsupported("column constraint not supported");
          break;
        }
        st.columns.push_back(std::move(cd));
      }
      if (cur().text == ",") { ++i_; continue; }
      break;
    }
    expect(")");
    if (!pk_seen) syntax("table requires a BIGINT primary key");
    if (vec_seen > 1) unsupported("at most one vector column per table");
    int pk_count = 0;
    for (auto& c : st.columns) if (c.primary) { ++pk_count; if (c.type != ColType::BigInt) unsupported("primary key must be BIGINT"); }
    if (pk_count != 1) unsupported("exactly one primary key column is required");
  }

  int64_t number_int() {
    if (cur().kind != Tok::Number) syntax("expected number");
    std::string_view t = cur().text;
    if (t.find('.') != std::string_view::npos || t.find('e') != std::string_view::npos ||
        t.find('E') != std::string_view::npos)
      syntax("expected integer");
    ++i_;
    int64_t v = 0;
    auto r = std::from_chars(t.data(), t.data() + t.size(), v);
    if (r.ec == std::errc::result_out_of_range) throw Error("value_out_of_range", "integer out of range");
    if (r.ec != std::errc() || r.ptr != t.data() + t.size()) syntax("expected integer");
    return v;
  }

  void insert(Statement& st) {
    expect_kw("INTO");
    st.table = ident();
    if (cur().text == "(") {
      ++i_;
      for (;;) {
        st.insert_columns.push_back(ident());
        if (cur().text == ",") { ++i_; continue; }
        break;
      }
      expect(")");
    }
    expect_kw("VALUES");
    for (;;) {
      expect("(");
      std::vector<ExprPtr> row;
      for (;;) {
        row.push_back(primary());
        if (cur().text == ",") { ++i_; continue; }
        break;
      }
      expect(")");
      st.rows.push_back(std::move(row));
      if (cur().text == ",") { ++i_; continue; }
      break;
    }
    if (cur().kind != Tok::End && cur().text != ";") unsupported("INSERT form not supported");
  }

  void select(Statement& st) {
    if (cur().upper == "DISTINCT") unsupported("DISTINCT is not supported");
    for (;;) {
      SelectItem it;
      if (cur().text == "*") { ++i_; it.star = true; }
      else {
        it.expr = or_expr();
        if (cur().upper == "AS") { ++i_; it.alias = ident(); }
        else if (cur().kind == Tok::Ident && cur().upper != "FROM") { it.alias = std::string(cur().text); ++i_; }
      }
      st.items.push_back(std::move(it));
      if (cur().text == ",") { ++i_; continue; }
      break;
    }
    expect_kw("FROM");
    if (cur().text == "(") unsupported("subqueries are not supported");
    st.table = ident();
    if (cur().text == "," || cur().upper == "JOIN" || cur().upper == "INNER" || cur().upper == "LEFT" ||
        cur().upper == "RIGHT" || cur().upper == "CROSS") unsupported("JOIN is not supported");
    if (cur().kind == Tok::Ident && cur().upper != "WHERE" && cur().upper != "ORDER" && cur().upper != "LIMIT") {
      st.table = st.table;  // bare alias, ignore
      ++i_;
    }
    if (cur().upper == "WHERE") { ++i_; st.where = or_expr(); }
    if (cur().upper == "GROUP" || cur().upper == "HAVING") unsupported("GROUP BY / HAVING are not supported");
    if (cur().upper == "ORDER") {
      ++i_; expect_kw("BY");
      for (;;) {
        OrderKey k;
        if (cur().kind != Tok::Ident) syntax("expected ORDER BY key");
        k.name = std::string(cur().text);
        ++i_;
        if (cur().text == ".") { ++i_; k.name = ident(); }
        if (cur().upper == "DESC") { k.desc = true; ++i_; }
        else if (cur().upper == "ASC") { ++i_; }
        st.order.push_back(std::move(k));
        if (cur().text == ",") { ++i_; continue; }
        break;
      }
    }
    if (cur().upper == "LIMIT") {
      ++i_;
      st.limit = number_int();
      st.has_limit = true;
      if (st.limit < 0) syntax("LIMIT must be non-negative");
      if (cur().text == "," || cur().upper == "OFFSET") unsupported("LIMIT offset is not supported");
    }
    if (cur().upper == "FOR" || cur().upper == "LOCK") unsupported("locking reads are not supported");
  }

  void update(Statement& st) {
    st.table = ident();
    expect_kw("SET");
    for (;;) {
      std::string col = ident();
      expect("=");
      ExprPtr e = or_expr();
      st.sets.emplace_back(std::move(col), std::move(e));
      if (cur().text == ",") { ++i_; continue; }
      break;
    }
    if (cur().upper == "WHERE") { ++i_; st.where = or_expr(); }
    if (cur().upper == "ORDER" || cur().upper == "LIMIT") unsupported("UPDATE ORDER BY / LIMIT not supported");
  }

  // OR is outside the agreed subset; detect it so we can report unsupported_sql.
  ExprPtr or_expr() {
    ExprPtr l = and_expr();
    if (cur().upper == "OR") unsupported("OR is not supported");
    return l;
  }

  ExprPtr and_expr() {
    ExprPtr l = cmp_expr();
    while (cur().upper == "AND") {
      ++i_;
      auto e = std::make_shared<Expr>();
      e->kind = ExprKind::And;
      e->lhs = l;
      e->rhs = cmp_expr();
      l = e;
    }
    return l;
  }

  ExprPtr cmp_expr() {
    ExprPtr l = primary();
    const std::string_view op = cur().text;
    CmpOp o;
    if (op == "=") o = CmpOp::Eq;
    else if (op == "<") o = CmpOp::Lt;
    else if (op == "<=") o = CmpOp::Le;
    else if (op == ">") o = CmpOp::Gt;
    else if (op == ">=") o = CmpOp::Ge;
    else if (op == "!=" || op == "<>") unsupported("only =, <, <=, >, >= are supported");
    else return l;
    ++i_;
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Cmp;
    e->cmp = o;
    e->lhs = l;
    e->rhs = primary();
    return e;
  }

  ExprPtr primary() {
    const Token& t = cur();
    if (t.kind == Tok::Number) {
      auto e = std::make_shared<Expr>();
      std::string_view s = t.text;
      if (s.find('.') == std::string_view::npos && s.find('e') == std::string_view::npos &&
          s.find('E') == std::string_view::npos) {
        e->kind = ExprKind::IntLit;
        int64_t v = 0;
        auto r = std::from_chars(s.data(), s.data() + s.size(), v);
        if (r.ec == std::errc::result_out_of_range) throw Error("value_out_of_range", "integer out of range");
        if (r.ec != std::errc() || r.ptr != s.data() + s.size()) syntax("bad integer literal");
        e->ival = v;
      } else {
        e->kind = ExprKind::StrLit;  // numeric literal kept as text; typed at execution
        e->sval = std::string(s);
      }
      ++i_;
      return e;
    }
    if (t.kind == Tok::String) {
      auto e = std::make_shared<Expr>();
      e->kind = ExprKind::StrLit;
      e->sval = t.has_decoded ? t.decoded : std::string(t.text);
      ++i_;
      return e;
    }
    if (t.text == "(") {
      ++i_;
      ExprPtr e = or_expr();
      if (cur().text == ")") { ++i_; return e; }
      unsupported("row constructors / subqueries are not supported");
    }
    if (t.upper == "NULL") unsupported("NULL is not supported");
    if (t.upper == "NOT" || t.upper == "LIKE" || t.upper == "IN" || t.upper == "BETWEEN" || t.upper == "IS")
      unsupported("predicate not supported");
    if (t.kind == Tok::Ident) {
      std::string name(t.text);
      std::string up = t.upper;
      ++i_;
      if (cur().text == ".") { ++i_; name = ident(); }
      if (cur().text == "(") {
        ++i_;
        if (up == "L2_DISTANCE") {
          auto e = std::make_shared<Expr>();
          e->kind = ExprKind::L2;
          e->lhs = or_expr();
          expect(",");
          e->rhs = or_expr();
          expect(")");
          return e;
        }
        unsupported("function not supported: " + name);
      }
      if (up == "COUNT" || up == "SUM" || up == "AVG" || up == "MIN" || up == "MAX")
        unsupported("aggregates are not supported");
      auto e = std::make_shared<Expr>();
      e->kind = ExprKind::Column;
      e->sval = name;
      return e;
    }
    syntax("unexpected token in expression");
  }
};

inline Statement parse(const std::string& text) { return Parser(text).parse(); }

}  // namespace sql