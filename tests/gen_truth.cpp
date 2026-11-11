// Deterministic data + brute-force truth generator for MiniVectorDB tests.
//   gen_truth <n> <dim> <nq> <sql_out> <truth_out>
// Emits CREATE TABLE + batched INSERTs, and the exact TopK (K=10) answer for
// each query both unfiltered and with an equality filter, ties by primary key.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static uint64_t st = 0x243F6A8885A308D3ull;
static uint64_t nxt() {
  st ^= st << 13; st ^= st >> 7; st ^= st << 17;
  return st;
}
static float unif() { return float(int64_t(nxt() >> 11) % 2000001 - 1000000) / 1000000.0f; }

int main(int argc, char** argv) {
  if (argc < 6) { std::fprintf(stderr, "usage: gen_truth n dim nq sql_out truth_out\n"); return 2; }
  int n = std::atoi(argv[1]), dim = std::atoi(argv[2]), nq = std::atoi(argv[3]);
  const int K = 10, BATCH = 100, NCAT = 7;
  std::vector<float> v(size_t(n) * dim);
  std::vector<int> cat(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    cat[size_t(i)] = i % NCAT;
    for (int d = 0; d < dim; ++d) v[size_t(i) * dim + d] = unif();
  }
  std::vector<float> q(size_t(nq) * dim);
  for (int i = 0; i < nq * dim; ++i) q[size_t(i)] = unif();

  FILE* s = std::fopen(argv[4], "wb");
  std::fprintf(s, "CREATE TABLE big (id BIGINT PRIMARY KEY, cat BIGINT, v VECTOR(%d));\n", dim);
  char buf[64];
  for (int start = 0; start < n; start += BATCH) {
    int stop = std::min(start + BATCH, n);
    std::string line = "INSERT INTO big VALUES ";
    for (int i = start; i < stop; ++i) {
      if (i != start) line += ',';
      line += '(' + std::to_string(i + 1) + ',' + std::to_string(cat[size_t(i)]) + ",'[";
      for (int d = 0; d < dim; ++d) {
        if (d) line += ',';
        std::snprintf(buf, sizeof buf, "%.9g", double(v[size_t(i) * dim + d]));
        line += buf;
      }
      line += "]')";
    }
    line += ";\n";
    std::fwrite(line.data(), 1, line.size(), s);
  }
  std::fclose(s);

  FILE* t = std::fopen(argv[5], "wb");
  std::fprintf(t, "[\n");
  bool first = true;
  std::vector<std::pair<double, int>> scored;
  for (int qi = 0; qi < nq; ++qi) {
    for (int pass = 0; pass < 2; ++pass) {
      bool use_filter = pass == 1;
      int want_cat = qi % NCAT;
      scored.clear();
      for (int i = 0; i < n; ++i) {
        if (use_filter && cat[size_t(i)] != want_cat) continue;
        double acc = 0;
        for (int d = 0; d < dim; ++d) {
          double diff = double(v[size_t(i) * dim + d]) - double(q[size_t(qi) * dim + d]);
          acc += diff * diff;
        }
        scored.emplace_back(std::sqrt(acc), i + 1);
      }
      std::sort(scored.begin(), scored.end(), [](const std::pair<double, int>& a,
                                                 const std::pair<double, int>& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
      });
      std::string sql = "SELECT id, L2_DISTANCE(v,'[";
      for (int d = 0; d < dim; ++d) {
        if (d) sql += ',';
        std::snprintf(buf, sizeof buf, "%.9g", double(q[size_t(qi) * dim + d]));
        sql += buf;
      }
      sql += "]') AS d FROM big ";
      if (use_filter) sql += "WHERE cat = " + std::to_string(want_cat) + " ";
      sql += "ORDER BY d LIMIT " + std::to_string(K) + ";";
      if (!first) std::fprintf(t, ",\n");
      first = false;
      std::fprintf(t, "{\"sql\":\"%s\",\"ids\":[", sql.c_str());
      for (int k = 0; k < K && k < int(scored.size()); ++k)
        std::fprintf(t, "%s%d", k ? "," : "", scored[size_t(k)].second);
      std::fprintf(t, "]}");
    }
  }
  std::fprintf(t, "\n]\n");
  std::fclose(t);
  std::fprintf(stderr, "generated n=%d dim=%d nq=%d\n", n, dim, nq);
  return 0;
}