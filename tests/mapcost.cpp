// Measure what a std::map lookup actually costs: tree depth, hop count, and
// how far apart the nodes sit in memory. Facts for the ablation explanation.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// std::map's node layout is implementation-defined; libstdc++'s _Rb_tree_node_base
// is {color, parent, left, right} followed by the value. We only need the child
// pointers to walk depth, so reinterpret the first 4 words.
struct NodeBase { uintptr_t color; uintptr_t parent; uintptr_t left; uintptr_t right; };

int main() {
  const int N = 10000;
  std::map<int64_t, size_t> m;
  for (int i = 0; i < N; ++i) m[int64_t(i + 1)] = size_t(i) * 528;

  // Recover the root via begin()'s leftmost node walking up to parent == header.
  // Simpler: measure depth by walking from each node's address is not portable,
  // so instead measure the observable quantities: lookup cost and node spread.
  std::vector<uintptr_t> addrs;
  addrs.reserve(N);
  for (auto& kv : m) addrs.push_back(reinterpret_cast<uintptr_t>(&kv));
  std::sort(addrs.begin(), addrs.end());
  uintptr_t span = addrs.back() - addrs.front();
  size_t gaps = 0;
  for (size_t i = 1; i < addrs.size(); ++i) if (addrs[i] - addrs[i - 1] > 64) ++gaps;

  // Sequential iteration cost (what the engine's hot loop does).
  volatile size_t sink = 0;
  double t0 = now();
  for (int rep = 0; rep < 200; ++rep)
    for (auto& kv : m) sink += kv.second;
  double t1 = now();
  double iter_ns = (t1 - t0) * 1e9 / (200.0 * N);

  // Random find cost (what the ablation's mode2 does).
  std::vector<int64_t> keys(N);
  for (int i = 0; i < N; ++i) keys[i] = int64_t(i + 1);
  uint64_t st = 12345;
  for (int i = N - 1; i > 0; --i) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; std::swap(keys[i], keys[st % (i + 1)]); }
  t0 = now();
  for (int rep = 0; rep < 200; ++rep)
    for (int i = 0; i < N; ++i) sink += m.find(keys[i])->second;
  t1 = now();
  double find_ns = (t1 - t0) * 1e9 / (200.0 * N);

  // Contiguous array scan of the same payload, for comparison.
  std::vector<size_t> flat(N);
  for (int i = 0; i < N; ++i) flat[i] = size_t(i) * 528;
  t0 = now();
  for (int rep = 0; rep < 200; ++rep)
    for (int i = 0; i < N; ++i) sink += flat[i];
  t1 = now();
  double array_ns = (t1 - t0) * 1e9 / (200.0 * N);

  std::printf("N                      = %d\n", N);
  std::printf("log2(N)                = %.2f   (RB-tree height bound 2*log2(N+1) = %.1f)\n",
              std::log2(double(N)), 2 * std::log2(double(N + 1)));
  std::printf("node address span      = %.1f KiB across %d nodes\n", span / 1024.0, N);
  std::printf("adjacent-in-order nodes farther than 64B apart = %zu / %d (%.1f%%)\n",
              gaps, N - 1, 100.0 * gaps / (N - 1));
  std::printf("sequential map iterate = %.1f ns/entry\n", iter_ns);
  std::printf("random map::find       = %.1f ns/lookup\n", find_ns);
  std::printf("contiguous array scan  = %.2f ns/entry\n", array_ns);
  std::printf("iterate / array        = %.1fx\n", iter_ns / array_ns);
  std::printf("sink=%zu\n", size_t(sink));
  return 0;
}