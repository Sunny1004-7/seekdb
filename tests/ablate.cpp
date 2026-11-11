// Ablation of the vector TopK hot loop, to separate arithmetic from bookkeeping.
//   mode0: distance only, contiguous float array
//   mode1: + per-row schema offset walk (BIGINT/BIGINT/VECTOR layout)
//   mode2: + per-row std::map lookup (stands in for the block cache)
//   mode3: + bounded top-K maintenance (the full hot loop shape)
// Same 10000 x 128 data, same 1000 queries, same 8-way float accumulation.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

static const int N = 10000, DIM = 128, NQ = 1000, K = 10;
static uint64_t st = 0x243F6A8885A308D3ull;
static uint64_t nxt() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static float unif() { return float(int64_t(nxt() >> 11) % 2000001 - 1000000) / 1000000.0f; }

static inline double l2(const float* ap, const float* bp) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int k = 0, n8 = DIM & ~7;
  for (; k < n8; k += 8) {
    float d0 = ap[k] - bp[k], d1 = ap[k + 1] - bp[k + 1];
    float d2 = ap[k + 2] - bp[k + 2], d3 = ap[k + 3] - bp[k + 3];
    float d4 = ap[k + 4] - bp[k + 4], d5 = ap[k + 5] - bp[k + 5];
    float d6 = ap[k + 6] - bp[k + 6], d7 = ap[k + 7] - bp[k + 7];
    s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
    s4 += d4 * d4; s5 += d5 * d5; s6 += d6 * d6; s7 += d7 * d7;
  }
  double acc = double((s0 + s1) + (s2 + s3)) + double((s4 + s5) + (s6 + s7));
  for (; k < DIM; ++k) { double d = double(ap[k]) - double(bp[k]); acc += d * d; }
  return std::sqrt(acc);
}

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Cand { double d; int64_t pk; };

int main() {
  // record layout mirrors the engine: [pk:8][cat:8][vector: DIM*4]
  const size_t REC = 8 + 8 + size_t(DIM) * 4;
  std::vector<char> recs(size_t(N) * REC);
  std::vector<float> flat(size_t(N) * DIM);
  std::map<int64_t, size_t> index;
  for (int i = 0; i < N; ++i) {
    int64_t pk = i + 1;
    char* r = recs.data() + size_t(i) * REC;
    std::memcpy(r, &pk, 8);
    int64_t cat = i % 7;
    std::memcpy(r + 8, &cat, 8);
    for (int d = 0; d < DIM; ++d) {
      float x = unif();
      flat[size_t(i) * DIM + d] = x;
      std::memcpy(r + 16 + size_t(d) * 4, &x, 4);
    }
    index[pk] = size_t(i) * REC;
  }
  std::vector<float> qs(size_t(NQ) * DIM);
  for (auto& x : qs) x = unif();

  volatile double sink = 0;
  for (int mode = 0; mode < 4; ++mode) {
    // warm
    for (int q = 0; q < 20; ++q) sink += l2(flat.data(), qs.data() + size_t(q) * DIM);
    double t0 = now();
    double acc_sum = 0;
    for (int q = 0; q < NQ; ++q) {
      const float* bp = qs.data() + size_t(q) * DIM;
      std::vector<Cand> top;
      top.reserve(K + 1);
      for (int i = 0; i < N; ++i) {
        const float* ap;
        if (mode == 0) {
          ap = flat.data() + size_t(i) * DIM;
        } else {
          size_t off;
          if (mode >= 2) {
            auto it = index.find(int64_t(i + 1));   // per-row map lookup
            off = it->second;
          } else {
            off = size_t(i) * REC;
          }
          const char* r = recs.data() + off;
          // per-row schema offset walk (mode >= 1)
          const char* p = r + 8;
          p += 8;                    // cat BIGINT
          ap = reinterpret_cast<const float*>(p);
        }
        double d = l2(ap, bp);
        if (mode < 3) { acc_sum += d; continue; }
        Cand c{d, int64_t(i + 1)};
        if (top.size() < K) {
          top.push_back(c);
          std::sort(top.begin(), top.end(), [](const Cand& a, const Cand& b) {
            return a.d != b.d ? a.d < b.d : a.pk < b.pk; });
        } else if (c.d < top.back().d || (c.d == top.back().d && c.pk < top.back().pk)) {
          top.back() = c;
          std::sort(top.begin(), top.end(), [](const Cand& a, const Cand& b) {
            return a.d != b.d ? a.d < b.d : a.pk < b.pk; });
        }
      }
      if (mode == 3) acc_sum += top.empty() ? 0 : top[0].d;
    }
    double t1 = now();
    sink += acc_sum;
    const char* name[] = {"mode0 distance only (contiguous)",
                          "mode1 + per-row offset walk",
                          "mode2 + per-row std::map lookup",
                          "mode3 + bounded top-K (full shape)"};
    std::printf("%-38s %8.3f ms/query   (total %.3f s)\n", name[mode],
                (t1 - t0) * 1000.0 / NQ, t1 - t0);
  }
  std::printf("sink=%g\n", double(sink));
  return 0;
}