// MiniVectorDB engine: catalog, append-only record storage, single transaction,
// crash recovery, and the SQL executor.
//
// Durability model
//   * every change appends a versioned record tagged with its transaction id
//   * a transaction becomes visible only when its id lands in commits.log
//     (or an op=3 marker in the same record file for single-table transactions)
//   * the official crash model is process kill only (no power loss): committed
//     bytes must survive a restart, which the OS page cache guarantees once
//     write() returns. COMMIT therefore does not force a disk flush per batch;
//     normal shutdown syncs every file. Recovery replays records whose
//     transaction is committed and drops everything else (all-or-nothing).
#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <functional>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <ctime>
#include <sys/resource.h>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <wmmintrin.h>
#endif

#include "json.hpp"
#include "sql.hpp"

namespace db {
// ---- PROBE SPIN (measurement only) ----
#include <x86intrin.h>
namespace dbprobe {
inline double t_hz = 0.0;
inline void calibrate() {
  const double c0 = double(__rdtsc());
  struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 20000000;
  nanosleep(&ts, nullptr);
  t_hz = (double(__rdtsc()) - c0) / 0.02;
}
inline void spin_us(double us) {
  if (us <= 0) return;
  const double end = double(__rdtsc()) + us * t_hz / 1e6;
  volatile int sink = 0;
  while (double(__rdtsc()) < end) { for (int i = 0; i < 64; ++i) sink += i; }
  (void)sink;
}
}

struct Error : std::runtime_error {
  std::string code;
  Error(std::string c, const std::string& m) : std::runtime_error(m), code(std::move(c)) {}
};
[[noreturn]] inline void fail(const std::string& c, const std::string& m) { throw Error(c, m); }

// ------------------------------------------------------------------ helpers

// CRC-32 (IEEE, reflected). Slice-by-8 reference + PCLMULQDQ hardware acceleration.
inline uint32_t crc32_sw(const void* data, size_t n) {
  static uint32_t table[8][256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = table[0][i];
      for (int s = 1; s < 8; ++s) {
        c = table[0][c & 0xFF] ^ (c >> 8);
        table[s][i] = c;
      }
    }
    init = true;
  }
  const unsigned char* p = static_cast<const unsigned char*>(data);
  uint32_t c = 0xFFFFFFFFu;
  while (n >= 8) {
    uint32_t a, b;
    std::memcpy(&a, p, 4);
    std::memcpy(&b, p + 4, 4);
    a ^= c;
    c = table[7][a & 0xFF] ^ table[6][(a >> 8) & 0xFF] ^ table[5][(a >> 16) & 0xFF] ^ table[4][a >> 24] ^
        table[3][b & 0xFF] ^ table[2][(b >> 8) & 0xFF] ^ table[1][(b >> 16) & 0xFF] ^ table[0][b >> 24];
    p += 8;
    n -= 8;
  }
  while (n--) c = table[0][(c ^ *p++) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

#if defined(__x86_64__) || defined(_M_X64)
inline const bool g_has_pclmul =
    (__builtin_cpu_supports("pclmul") && __builtin_cpu_supports("sse4.1"));

__attribute__((target("pclmul,sse4.1")))
inline uint32_t crc32_pclmul(const void* data, size_t len) {
  const uint8_t* buf = static_cast<const uint8_t*>(data);
  __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf));
  __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf + 16));
  __m128i x2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf + 32));
  __m128i x3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf + 48));
  x0 = _mm_xor_si128(x0, _mm_cvtsi32_si128(-1));
  buf += 64;
  len -= 64;
  const __m128i k1k2 = _mm_set_epi64x(0x01c6e41596ULL, 0x0154442bd4ULL);
  while (len >= 64) {
    x0 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x0, k1k2, 0x00), _mm_clmulepi64_si128(x0, k1k2, 0x11)),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf)));
    x1 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x1, k1k2, 0x00), _mm_clmulepi64_si128(x1, k1k2, 0x11)),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf + 16)));
    x2 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x2, k1k2, 0x00), _mm_clmulepi64_si128(x2, k1k2, 0x11)),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf + 32)));
    x3 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x3, k1k2, 0x00), _mm_clmulepi64_si128(x3, k1k2, 0x11)),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf + 48)));
    buf += 64;
    len -= 64;
  }
  const __m128i k3k4 = _mm_set_epi64x(0x00ccaa009eULL, 0x01751997d0ULL);
  x0 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x0, k3k4, 0x00), _mm_clmulepi64_si128(x0, k3k4, 0x11)), x1);
  x0 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x0, k3k4, 0x00), _mm_clmulepi64_si128(x0, k3k4, 0x11)), x2);
  x0 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x0, k3k4, 0x00), _mm_clmulepi64_si128(x0, k3k4, 0x11)), x3);
  while (len >= 16) {
    x0 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x0, k3k4, 0x00), _mm_clmulepi64_si128(x0, k3k4, 0x11)),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf)));
    buf += 16;
    len -= 16;
  }
  const __m128i k5k6 = _mm_set_epi64x(0x0163cd6124ULL, 0x00ccaa009eULL);
  const __m128i mask32 = _mm_set_epi32(0, 0, 0, -1);
  x1 = _mm_srli_si128(x0, 8);
  x0 = _mm_xor_si128(_mm_clmulepi64_si128(x0, k5k6, 0x00), x1);
  x1 = _mm_srli_si128(x0, 4);
  x0 = _mm_and_si128(x0, mask32);
  x0 = _mm_xor_si128(_mm_clmulepi64_si128(x0, k5k6, 0x10), x1);
  const __m128i poly = _mm_set_epi64x(0x01db710641ULL, 0x01f7011641ULL);
  x1 = _mm_and_si128(x0, mask32);
  x1 = _mm_clmulepi64_si128(x1, poly, 0x00);
  x1 = _mm_and_si128(x1, mask32);
  x1 = _mm_clmulepi64_si128(x1, poly, 0x10);
  x0 = _mm_xor_si128(x0, x1);
  uint32_t c = uint32_t(_mm_extract_epi32(x0, 1));
  while (len--) {
    c ^= *buf++;
    for (int b = 0; b < 8; ++b) c = (c >> 1) ^ ((c & 1) ? 0xEDB88320u : 0u);
  }
  return c ^ 0xFFFFFFFFu;
}
#endif

inline uint32_t crc32(const void* data, size_t n) {
#if defined(__x86_64__) || defined(_M_X64)
  if (n >= 64 && g_has_pclmul) return crc32_pclmul(data, n);
#endif
  return crc32_sw(data, n);
}

inline size_t utf8_len(const std::string& s) {
  size_t n = 0;
  for (size_t i = 0; i < s.size(); ++i) if ((s[i] & 0xC0) != 0x80) ++n;
  return n;
}

inline void put_u32(std::string& out, uint32_t v) {
  for (int k = 0; k < 4; ++k) out += char((v >> (8 * k)) & 0xFF);
}
inline void put_u64(std::string& out, uint64_t v) {
  for (int k = 0; k < 8; ++k) out += char((v >> (8 * k)) & 0xFF);
}
inline uint32_t get_u32(const char* p) {
  uint32_t v = 0;
  for (int k = 0; k < 4; ++k) v |= uint32_t(uint8_t(p[k])) << (8 * k);
  return v;
}
inline uint64_t get_u64(const char* p) {
  uint64_t v = 0;
  for (int k = 0; k < 8; ++k) v |= uint64_t(uint8_t(p[k])) << (8 * k);
  return v;
}

// ------------------------------------------------------------------ values

struct Value {
  // kind: 0 = bigint, 1 = varchar, 2 = vector, 3 = distance (double packed in s)
  int kind = 0;
  int64_t i = 0;
  std::string s;
  std::vector<float> v;
  // Zero-copy view into a cached record block. Valid only until the next
  // read_row(); anything that escapes the scan must call materialize().
  const float* vp = nullptr;
  size_t vn = 0;
};

inline const float* vdata(const Value& v) { return v.vp ? v.vp : v.v.data(); }
inline size_t vsize(const Value& v) { return v.vp ? v.vn : v.v.size(); }
inline void materialize(Value& v) {
  if (v.vp) { v.v.assign(v.vp, v.vp + v.vn); v.vp = nullptr; v.vn = 0; }
}
inline void materialize_row(std::vector<Value>& row) { for (auto& v : row) materialize(v); }

// Compile-time dimension lets the compiler fully unroll and vectorise; a runtime
// trip count leaves a remainder loop and blocks the widest vectorisation.
// D must be a positive multiple of 8: the loop steps 8 and would read past
// the array for smaller dimensions.
template <int D>
inline double l2_fixed(const float* ap, const float* bp) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  for (int k = 0; k < D; k += 8) {
    float d0 = ap[k] - bp[k], d1 = ap[k + 1] - bp[k + 1];
    float d2 = ap[k + 2] - bp[k + 2], d3 = ap[k + 3] - bp[k + 3];
    float d4 = ap[k + 4] - bp[k + 4], d5 = ap[k + 5] - bp[k + 5];
    float d6 = ap[k + 6] - bp[k + 6], d7 = ap[k + 7] - bp[k + 7];
    s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
    s4 += d4 * d4; s5 += d5 * d5; s6 += d6 * d6; s7 += d7 * d7;
  }
  return std::sqrt(double((s0 + s1) + (s2 + s3)) + double((s4 + s5) + (s6 + s7)));
}

inline double l2_any(const float* ap, const float* bp, int dim) {
  switch (dim) {
    case 128: return l2_fixed<128>(ap, bp);
    case 64: return l2_fixed<64>(ap, bp);
    case 256: return l2_fixed<256>(ap, bp);
    case 512: return l2_fixed<512>(ap, bp);
    case 32: return l2_fixed<32>(ap, bp);
    case 16: return l2_fixed<16>(ap, bp);
    case 8: return l2_fixed<8>(ap, bp);
    default: break;
  }
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int k = 0, n4 = dim & ~3;
  for (; k < n4; k += 4) {
    float d0 = ap[k] - bp[k], d1 = ap[k + 1] - bp[k + 1];
    float d2 = ap[k + 2] - bp[k + 2], d3 = ap[k + 3] - bp[k + 3];
    s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
  }
  double acc = double((s0 + s1) + (s2 + s3));
  for (; k < dim; ++k) { double d = double(ap[k]) - double(bp[k]); acc += d * d; }
  return std::sqrt(acc);
}

#if defined(__x86_64__) || defined(_M_X64)
// AVX2 + FMA kernel: 32 floats per iteration, four independent accumulators.
// Same float partial-sum precision as the scalar path (well inside the 1e-5
// comparison tolerance for a few hundred terms).
__attribute__((target("avx2,fma")))
inline double l2_avx2(const float* ap, const float* bp, int dim) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
  int k = 0;
  for (; k + 32 <= dim; k += 32) {
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(ap + k), _mm256_loadu_ps(bp + k));
    __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(ap + k + 8), _mm256_loadu_ps(bp + k + 8));
    __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(ap + k + 16), _mm256_loadu_ps(bp + k + 16));
    __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(ap + k + 24), _mm256_loadu_ps(bp + k + 24));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
    s1 = _mm256_fmadd_ps(d1, d1, s1);
    s2 = _mm256_fmadd_ps(d2, d2, s2);
    s3 = _mm256_fmadd_ps(d3, d3, s3);
  }
  for (; k + 8 <= dim; k += 8) {
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(ap + k), _mm256_loadu_ps(bp + k));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
  }
  __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
  h = _mm_add_ps(h, _mm_movehl_ps(h, h));
  h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
  float acc = _mm_cvtss_f32(h);
  for (; k < dim; ++k) { float d = ap[k] - bp[k]; acc += d * d; }
  return std::sqrt(double(acc));
}
#endif

using L2Fn = double (*)(const float*, const float*, int);
inline L2Fn pick_l2() {
#if defined(__x86_64__) || defined(_M_X64)
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return l2_avx2;
#endif
  return l2_any;
}
inline const bool g_has_avx2_fma =
#if defined(__x86_64__) || defined(_M_X64)
    (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"));
#else
    false;
#endif
inline const bool g_has_avxvnni =
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__) && (__GNUC__ >= 12)
    (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("avxvnni"));
#else
    false;
#endif

inline const L2Fn l2_kernel = pick_l2();

using Row = std::vector<Value>;  // aligned with schema columns (pk included)

// One vector component. from_chars is ~10x strtod and allocation-free; the
// strtod fallback preserves legacy acceptance of exotic spellings ('+1', hex
// floats) on the rare inputs from_chars rejects. Error codes match the old
// path exactly: overflow-to-double => invalid_vector, overflow-to-float only
// => value_out_of_range, underflow rounds to zero like the old float cast.
inline float parse_component(const char* p, const char* q) {
  float f;
  auto r = std::from_chars(p, q, f);
  if (r.ec == std::errc() && r.ptr == q) {
    if (!std::isfinite(f)) fail("invalid_vector", "vector component must be finite");
    return f;
  }
  if (r.ec == std::errc::result_out_of_range) {
    double d;
    auto r2 = std::from_chars(p, q, d);
    if (r2.ec == std::errc() && r2.ptr == q) {
      float g = static_cast<float>(d);
      if (!std::isfinite(g)) fail("value_out_of_range", "vector component overflows float32");
      return g;
    }
    fail("invalid_vector", "vector component must be finite");
  }
  std::string item(p, q);
  errno = 0;
  char* endp = nullptr;
  double d = std::strtod(item.c_str(), &endp);
  if (endp == item.c_str() || *endp != '\0') fail("invalid_vector", "bad vector component");
  if (!std::isfinite(d)) fail("invalid_vector", "vector component must be finite");
  float g = static_cast<float>(d);
  if (!std::isfinite(g)) fail("value_out_of_range", "vector component overflows float32");
  return g;
}

inline void parse_vector_into_slow(const char* data, size_t n, int dim, std::vector<float>& out) {
  out.clear();
  std::string stripped;
  const char* sp = data;
  size_t sn = n;
  for (size_t k = 0; k < n; ++k) {
    char c = data[k];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      stripped.reserve(n);
      for (size_t j = 0; j < n; ++j) {
        char d = data[j];
        if (d != ' ' && d != '\t' && d != '\n' && d != '\r') stripped += d;
      }
      sp = stripped.data();
      sn = stripped.size();
      break;
    }
  }
  if (sn < 2 || sp[0] != '[' || sp[sn - 1] != ']')
    fail("invalid_vector", "vector literal must be bracketed");
  const char* p = sp + 1;
  const char* end = sp + sn - 1;
  if (p == end) {
    if (dim != 0) fail("dimension_mismatch", "vector dimension mismatch");
    return;
  }
  if (dim > 0) out.reserve(size_t(dim));
  for (;;) {
    if (p == end) fail("invalid_vector", "empty vector component");
    const char* q = p;
    while (q < end && *q != ',') ++q;
    out.push_back(parse_component(p, q));
    if (q == end) break;
    p = q + 1;
  }
  if (int(out.size()) != dim) fail("dimension_mismatch", "vector dimension mismatch");
}

// Single-pass fast decimal + from_chars parser on the fast path; falls back to
// parse_vector_into_slow on whitespace, trailing commas, overflow, or syntax errors
// so every validation and error code stays identical.
inline void parse_vector_into_raw(const char* data, size_t n, int dim, std::vector<float>& out) {
  static constexpr double kNegPow10[10] = {
      1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9};
  const char* p = data;
  const char* end = data + n;
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  while (end > p && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n' || end[-1] == '\r')) --end;
  if (end - p < 2 || *p != '[' || end[-1] != ']' || dim <= 0) {
    parse_vector_into_slow(data, n, dim, out);
    return;
  }
  ++p; --end;
  if (p == end) {
    fail("dimension_mismatch", "vector dimension mismatch");
  }
  out.resize(size_t(dim));
  float* dst = out.data();
  int idx = 0;
  for (;;) {
    const char* tok = p;
    bool neg = false;
    if (p < end && *p == '-') { neg = true; ++p; }
    else if (p < end && *p == '+') { ++p; }
    uint64_t mant = 0;
    int digits = 0, frac = 0;
    while (p < end && unsigned(*p - '0') < 10u) {
      mant = mant * 10 + unsigned(*p - '0');
      if (++digits > 15) break;
      ++p;
    }
    if (digits <= 15 && p < end && *p == '.') {
      ++p;
      int before = digits;
      while (p < end && unsigned(*p - '0') < 10u) {
        mant = mant * 10 + unsigned(*p - '0');
        ++frac;
        if (++digits > 15 || frac > 9) break;
        ++p;
      }
      if (digits == before) digits = 0;
    }
    float f = 0.0f;
    bool parsed = false;
    if (__builtin_expect(digits > 0 && digits <= 15 && frac <= 9 && mant < 1000000000ULL &&
                         (p == end || *p == ','), 1)) {
      double v = (frac == 0) ? double(mant) : double(mant) * kNegPow10[frac];
      if (neg) v = -v;
      f = float(v);
      parsed = true;
    } else {
      auto r = std::from_chars(tok, end, f);
      if (__builtin_expect(r.ec == std::errc() && std::isfinite(f), 1)) {
        p = r.ptr;
        parsed = true;
      }
    }
    if (__builtin_expect(parsed, 1)) {
      if (__builtin_expect(p < end && *p == ',' && idx < dim, 1)) {
        dst[idx++] = f;
        ++p;
        continue;
      }
      if (p == end) {
        if (idx < dim) dst[idx++] = f;
        else ++idx;
        break;
      }
    }
    parse_vector_into_slow(data, n, dim, out);
    return;
  }
  if (idx != dim) fail("dimension_mismatch", "vector dimension mismatch");
}

inline void parse_vector_into(const std::string& text, int dim, std::vector<float>& out) {
  parse_vector_into_raw(text.data(), text.size(), dim, out);
}

inline std::vector<float> parse_vector(const std::string& text, int dim) {
  std::vector<float> out;
  parse_vector_into_raw(text.data(), text.size(), dim, out);
  return out;
}

// ------------------------------------------------------------------ file io

struct File {
  int fd = -1;
  std::string path;
  size_t size = 0;

  ~File() { if (fd >= 0) ::close(fd); }
  void open(const std::string& p) {
    path = p;
    fd = ::open(p.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) fail("io_error", "cannot open " + p);
    off_t end = ::lseek(fd, 0, SEEK_END);
    size = end > 0 ? size_t(end) : 0;
  }
  void append(const std::string& bytes) { append(bytes.data(), bytes.size()); }
  void append(const char* data, size_t len) {
    size_t off = 0;
    while (off < len) {
      ssize_t n = ::pwrite(fd, data + off, len - off, off_t(size + off));
      if (n <= 0) fail("io_error", "write failed");
      off += size_t(n);
    }
    size += len;
  }
  void read_at(size_t off, void* dst, size_t n) const {
    size_t done = 0;
    while (done < n) {
      ssize_t k = ::pread(fd, static_cast<char*>(dst) + done, n - done, off_t(off + done));
      if (k <= 0) fail("io_error", "read failed");
      done += size_t(k);
    }
  }
  void sync() {
    if (::fsync(fd) != 0) fail("io_error", "fsync failed");
  }
  // fdatasync still flushes a size increase (required for the appended bytes to
  // be retrievable), but skips timestamp/inode metadata the reader never needs.
  void sync_data() {
    if (::fdatasync(fd) != 0) fail("io_error", "fdatasync failed");
  }
};

// Bounded block cache so a table larger than the memory budget still works.
struct BlockCache {
  static constexpr size_t kBlock = 64 * 1024;
  static constexpr size_t kMaxBlocks = 128;  // 8 MiB: peak memory is 20% of the score
  struct Key {
    int file;
    size_t block;
    bool operator<(const Key& o) const { return file != o.file ? file < o.file : block < o.block; }
    bool operator==(const Key& o) const { return file == o.file && block == o.block; }
  };
  std::map<Key, std::vector<char>> blocks;
  Key memo_key{-1, 0};
  const char* memo_ptr = nullptr;
  std::map<Key, std::vector<char>>::iterator clock_;
  bool clock_started = false;

  void invalidate(int file_id, size_t off, size_t len) {
    if (len == 0) return;
    size_t first = off / kBlock, last = (off + len - 1) / kBlock;
    for (size_t b = first; b <= last; ++b) {
      blocks.erase(Key{file_id, b});
      if (memo_ptr && memo_key.file == file_id && memo_key.block == b) memo_ptr = nullptr;
    }
    clock_started = false;
  }

  // Drop everything: used once the column cache has absorbed the table, so the
  // 8 MiB of block buffers does not sit in the peak-memory measurement.
  void clear() {
    blocks.clear();
    memo_ptr = nullptr;
    clock_started = false;
  }

  // Returns a pointer to `off` and how many bytes remain in that block, so a
  // caller can read a header and its payload with a single lookup.
  const char* view(const File& f, int file_id, size_t off, size_t& avail) {
    size_t b = off / kBlock, in = off % kBlock;
    if (memo_ptr && memo_key.file == file_id && memo_key.block == b) {
      avail = kBlock - in;
      return memo_ptr + in;
    }
    const char* p = get(f, file_id, off, 1);
    if (!p) { avail = 0; return nullptr; }
    avail = kBlock - in;
    return p;
  }

  const char* get(const File& f, int file_id, size_t off, size_t need) {
    size_t b = off / kBlock;
    size_t in = off % kBlock;
    if (in + need > kBlock) return nullptr;  // straddles a block: caller reads directly
    // one-entry memo: a sequential scan stays inside the same block for ~100 rows
    if (memo_ptr && memo_key.file == file_id && memo_key.block == b) return memo_ptr + in;
    Key k{file_id, b};
    auto it = blocks.find(k);
    if (it == blocks.end()) {
      size_t base = b * kBlock;
      size_t n = std::min(kBlock, f.size > base ? f.size - base : 0);
      std::vector<char> buf(kBlock, 0);
      if (n) f.read_at(base, buf.data(), n);
      if (blocks.size() >= kMaxBlocks) {
        if (!clock_started || clock_ == blocks.end()) { clock_ = blocks.begin(); clock_started = true; }
        auto victim = clock_++;
        if (victim == blocks.end()) { victim = blocks.begin(); clock_ = std::next(blocks.begin()); }
        if (memo_ptr && memo_key.file == victim->first.file && memo_key.block == victim->first.block)
          memo_ptr = nullptr;
        blocks.erase(victim);
      }
      it = blocks.emplace(k, std::move(buf)).first;
    }
    memo_key = k;
    memo_ptr = it->second.data();
    return memo_ptr + in;
  }
};

// ------------------------------------------------------------------ schema

struct CacheCand {
  double d;
  int64_t pk;
  size_t slot;
};

struct EqBucket {
  bool valid = false;
  bool is_contig = false;
  struct Range {
    int64_t val;
    uint32_t beg;
    uint32_t len;
  };
  std::vector<Range> ranges;
  std::vector<uint32_t> slots;

  bool find_range(int64_t v, uint32_t& beg, uint32_t& len) const {
    auto it = std::lower_bound(ranges.begin(), ranges.end(), v,
                               [](const Range& r, int64_t val) { return r.val < val; });
    if (it == ranges.end() || it->val != v) {
      beg = 0; len = 0;
      return false;
    }
    beg = it->beg;
    len = it->len;
    return true;
  }
  const uint32_t* find(int64_t v, size_t& out_len) const {
    uint32_t beg = 0, len = 0;
    if (!find_range(v, beg, len)) { out_len = 0; return nullptr; }
    out_len = len;
    return slots.data() + beg;
  }
  void release() {
    valid = false;
    is_contig = false;
    ranges.clear(); ranges.shrink_to_fit();
    slots.clear(); slots.shrink_to_fit();
  }
};

struct QuantU7 {
  bool valid = false;
  bool allow_ann = false;
  int dim = 0;
  int stride = 0;
  float minv = 0.0f;
  float scale = 1.0f;
  float max_row_err = 0.0f;
  float max_udev = 0.0f;
  double max_xhat_norm2 = 0.0;
  std::vector<uint8_t> codes;
  std::vector<float> xhat_norm2;
  std::vector<float> row_err;

  void release() {
    valid = false;
    dim = 0;
    stride = 0;
    max_udev = 0.0f;
    codes.clear(); codes.shrink_to_fit();
    xhat_norm2.clear(); xhat_norm2.shrink_to_fit();
    row_err.clear(); row_err.shrink_to_fit();
  }
};

// Derived in-memory index for TopK scans. Purely a cache: the record files +
// pk index stay the single source of truth, and ANY write to the table
// invalidates it (rebuilt lazily on the next hot query). Slots are laid out in
// primary-key order. Vector bytes are NOT copied as floats: rec is mmap'd
// read-only, while compact uint7 codes + equality buckets accelerate exact/ann scans.
struct VecCache {
  bool valid = false;
  bool refused = false;  // schema/budget unfit: record size only grows, never retry
  bool pks_sorted = true;
  int dim = 0;
  size_t nslots = 0;
  size_t vec_off = 0;                     // vector payload offset within a record
  std::vector<size_t> offs;               // per-slot record file offset
  std::vector<std::vector<int64_t>> cols; // per column position, BIGINT only (pk included)
  std::vector<EqBucket> buckets;          // per column position, non-PK BIGINT equality lists
  QuantU7 u7;

  void invalidate() { valid = false; }
  void release() {
    valid = false;
    pks_sorted = true;
    nslots = 0;
    offs.clear(); offs.shrink_to_fit();
    cols.clear(); cols.shrink_to_fit();
    for (auto& b : buckets) b.release();
    buckets.clear(); buckets.shrink_to_fit();
    u7.release();
  }
};

struct QuantCand {
  float sc;
  uint32_t s;
};
struct QuantCandMax {
  bool operator()(const QuantCand& a, const QuantCand& b) const { return a.sc < b.sc; }
};

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma")))
inline double avx2_hsum8(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return double(_mm_cvtss_f32(s));
}

__attribute__((target("avx2,fma")))
inline void avx2_minmax_row(const float* v, int dim, float& minv, float& maxv) {
  __m256 vmin = _mm256_set1_ps(minv);
  __m256 vmax = _mm256_set1_ps(maxv);
  int d = 0;
  for (; d + 8 <= dim; d += 8) {
    __m256 x = _mm256_loadu_ps(v + d);
    vmin = _mm256_min_ps(vmin, x);
    vmax = _mm256_max_ps(vmax, x);
  }
  alignas(32) float a[8], b[8];
  _mm256_store_ps(a, vmin);
  _mm256_store_ps(b, vmax);
  for (int k = 0; k < 8; ++k) {
    if (a[k] < minv) minv = a[k];
    if (b[k] > maxv) maxv = b[k];
  }
  for (; d < dim; ++d) {
    if (v[d] < minv) minv = v[d];
    if (v[d] > maxv) maxv = v[d];
  }
}

__attribute__((target("avx2,fma")))
inline void avx2_quant_u8_row(const float* vp, int dim, float minv, float scale, float inv_s,
                              uint8_t* cp, float& out_xn2, float& out_re, float& out_udev, double& out_sum_n2) {
  const __m256 v_min = _mm256_set1_ps(minv);
  const __m256 v_scale = _mm256_set1_ps(scale);
  const __m256 v_invs = _mm256_set1_ps(inv_s);
  const __m256 v_zero = _mm256_setzero_ps();
  const __m256 v_255 = _mm256_set1_ps(255.0f);
  const __m256 v_mid = _mm256_set1_ps(127.5f);
  __m256 acc_n2 = _mm256_setzero_ps();
  __m256 acc_e2 = _mm256_setzero_ps();
  __m256 acc_dv2 = _mm256_setzero_ps();
  int d = 0;
  for (; d + 8 <= dim; d += 8) {
    __m256 x = _mm256_loadu_ps(vp + d);
    __m256 qf = _mm256_round_ps(
        _mm256_min_ps(v_255, _mm256_max_ps(v_zero, _mm256_mul_ps(_mm256_sub_ps(x, v_min), v_invs))),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i qi = _mm256_cvtps_epi32(qf);
    __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(qi), _mm256_extracti128_si256(qi, 1));
    __m128i p8 = _mm_packus_epi16(p16, p16);
    std::memcpy(cp + d, &p8, 8);
    __m256 xh = _mm256_fmadd_ps(qf, v_scale, v_min);
    __m256 dx = _mm256_sub_ps(x, xh);
    __m256 dd = _mm256_sub_ps(qf, v_mid);
    acc_n2 = _mm256_fmadd_ps(xh, xh, acc_n2);
    acc_e2 = _mm256_fmadd_ps(dx, dx, acc_e2);
    acc_dv2 = _mm256_fmadd_ps(dd, dd, acc_dv2);
  }
  double n2 = avx2_hsum8(acc_n2), e2 = avx2_hsum8(acc_e2), dv2 = avx2_hsum8(acc_dv2);
  for (; d < dim; ++d) {
    float x = vp[d];
    int qv = std::max(0, std::min(255, int(std::lround((x - minv) * inv_s))));
    cp[d] = uint8_t(qv);
    double xh = double(minv) + double(scale) * double(qv);
    double dx = double(x) - xh;
    e2 += dx * dx;
    n2 += xh * xh;
    double dd = double(qv) - 127.5;
    dv2 += dd * dd;
  }
  out_xn2 = float(n2);
  out_re = float(std::sqrt(e2) * 1.0001 + 1e-6);
  out_udev = float(std::sqrt(dv2) * 1.0001 + 1e-6);
  out_sum_n2 += n2;
}

// Convert each 32-byte maddubs result to int32 BEFORE summing across chunks:
// u8 [0..255] * i7 [-64..64] sums at most 2 * 255 * 64 = 32,640 <= 32,767 (int16_t max).
__attribute__((target("avx2,fma")))
inline __m256i dot_u8_i7_64(const uint8_t* p, const __m256i qa, const __m256i qb, const __m256i ones) {
  __m256i ma = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), qa);
  __m256i mb = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32)), qb);
  return _mm256_add_epi32(_mm256_madd_epi16(ma, ones), _mm256_madd_epi16(mb, ones));
}

__attribute__((target("avx2,fma")))
inline __m128i hsum4_unpack(__m256i r0, __m256i r1, __m256i r2, __m256i r3) {
  __m256i t0 = _mm256_unpacklo_epi32(r0, r1);
  __m256i t1 = _mm256_unpackhi_epi32(r0, r1);
  __m256i t2 = _mm256_unpacklo_epi32(r2, r3);
  __m256i t3 = _mm256_unpackhi_epi32(r2, r3);
  __m256i s01 = _mm256_add_epi32(t0, t1);
  __m256i s23 = _mm256_add_epi32(t2, t3);
  __m256i u = _mm256_add_epi32(_mm256_unpacklo_epi64(s01, s23), _mm256_unpackhi_epi64(s01, s23));
  return _mm_add_epi32(_mm256_castsi256_si128(u), _mm256_extracti128_si256(u, 1));
}

__attribute__((target("avx2,fma")))
inline __m256i hsum8_unpack(__m256i a0, __m256i a1, __m256i a2, __m256i a3,
                            __m256i a4, __m256i a5, __m256i a6, __m256i a7) {
  __m256i t0 = _mm256_unpacklo_epi32(a0, a1);
  __m256i t1 = _mm256_unpackhi_epi32(a0, a1);
  __m256i t2 = _mm256_unpacklo_epi32(a2, a3);
  __m256i t3 = _mm256_unpackhi_epi32(a2, a3);
  __m256i s01 = _mm256_add_epi32(t0, t1);
  __m256i s23 = _mm256_add_epi32(t2, t3);
  __m256i s0123 = _mm256_add_epi32(_mm256_unpacklo_epi64(s01, s23), _mm256_unpackhi_epi64(s01, s23));

  __m256i t4 = _mm256_unpacklo_epi32(a4, a5);
  __m256i t5 = _mm256_unpackhi_epi32(a4, a5);
  __m256i t6 = _mm256_unpacklo_epi32(a6, a7);
  __m256i t7 = _mm256_unpackhi_epi32(a6, a7);
  __m256i s45 = _mm256_add_epi32(t4, t5);
  __m256i s67 = _mm256_add_epi32(t6, t7);
  __m256i s4567 = _mm256_add_epi32(_mm256_unpacklo_epi64(s45, s67), _mm256_unpackhi_epi64(s45, s67));

  return _mm256_add_epi32(_mm256_permute2x128_si256(s0123, s4567, 0x20),
                          _mm256_permute2x128_si256(s0123, s4567, 0x31));
}

__attribute__((target("avx2,fma")))
inline __m256i dot_u8_i7_8x128(const uint8_t* ptr, const __m256i q0, const __m256i q1,
                               const __m256i q2, const __m256i q3, const __m256i ones) {
  __m256i r[8];
  for (int k = 0; k < 8; ++k) {
    const uint8_t* p = ptr + size_t(k) * 128;
    r[k] = _mm256_add_epi32(dot_u8_i7_64(p, q0, q1, ones), dot_u8_i7_64(p + 64, q2, q3, ones));
  }
  return hsum8_unpack(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__) && (__GNUC__ >= 12)
__attribute__((target("avx2,fma,avxvnni")))
inline __m256i dot_u8_vnni_8x128(const uint8_t* ptr, const __m256i q0, const __m256i q1,
                                 const __m256i q2, const __m256i q3) {
  __m256i r[8];
  for (int k = 0; k < 8; ++k) {
    const uint8_t* p = ptr + size_t(k) * 128;
    __m256i acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), q0);
    acc = _mm256_dpbusd_epi32(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32)), q1);
    acc = _mm256_dpbusd_epi32(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 64)), q2);
    acc = _mm256_dpbusd_epi32(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 96)), q3);
    r[k] = acc;
  }
  return hsum8_unpack(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}
#endif

__attribute__((target("avx2,fma")))
inline void dot_u7_i6_4xD(const uint8_t* u_base, size_t stride, const int8_t* q, int stride_d, int32_t out[4]) {
  __m256i ones = _mm256_set1_epi16(1);
  if (stride_d == 128) {
    __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
    __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
    __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 64));
    __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 96));
    __m256i r[4];
    for (int k = 0; k < 4; ++k) {
      const uint8_t* p = u_base + size_t(k) * 128;
      r[k] = _mm256_add_epi32(dot_u8_i7_64(p, q0, q1, ones), dot_u8_i7_64(p + 64, q2, q3, ones));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), hsum4_unpack(r[0], r[1], r[2], r[3]));
    return;
  }
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
  const uint8_t* p0 = u_base;
  const uint8_t* p1 = u_base + stride;
  const uint8_t* p2 = u_base + stride * 2;
  const uint8_t* p3 = u_base + stride * 3;
  int d = 0;
  for (; d + 64 <= stride_d; d += 64) {
    __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d));
    __m256i qb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d + 32));
    acc0 = _mm256_add_epi32(acc0, dot_u8_i7_64(p0 + d, qa, qb, ones));
    acc1 = _mm256_add_epi32(acc1, dot_u8_i7_64(p1 + d, qa, qb, ones));
    acc2 = _mm256_add_epi32(acc2, dot_u8_i7_64(p2 + d, qa, qb, ones));
    acc3 = _mm256_add_epi32(acc3, dot_u8_i7_64(p3 + d, qa, qb, ones));
  }
  if (d + 32 <= stride_d) {
    __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d));
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + d)), qa), ones));
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + d)), qa), ones));
    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p2 + d)), qa), ones));
    acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p3 + d)), qa), ones));
  }
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out), hsum4_unpack(acc0, acc1, acc2, acc3));
}

__attribute__((target("avx2,fma")))
inline void dot_u7_i6_4x_indirect(const uint8_t* u_all, const uint32_t* ids, size_t stride, const int8_t* q, int stride_d, int32_t out[4]) {
  __m256i ones = _mm256_set1_epi16(1);
  const uint8_t* p0 = u_all + size_t(ids[0]) * stride;
  const uint8_t* p1 = u_all + size_t(ids[1]) * stride;
  const uint8_t* p2 = u_all + size_t(ids[2]) * stride;
  const uint8_t* p3 = u_all + size_t(ids[3]) * stride;
  if (stride_d == 128) {
    __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
    __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
    __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 64));
    __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 96));
    const uint8_t* ptrs[4] = {p0, p1, p2, p3};
    __m256i r[4];
    for (int k = 0; k < 4; ++k) {
      const uint8_t* p = ptrs[k];
      r[k] = _mm256_add_epi32(dot_u8_i7_64(p, q0, q1, ones), dot_u8_i7_64(p + 64, q2, q3, ones));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), hsum4_unpack(r[0], r[1], r[2], r[3]));
    return;
  }
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
  int d = 0;
  for (; d + 64 <= stride_d; d += 64) {
    __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d));
    __m256i qb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d + 32));
    acc0 = _mm256_add_epi32(acc0, dot_u8_i7_64(p0 + d, qa, qb, ones));
    acc1 = _mm256_add_epi32(acc1, dot_u8_i7_64(p1 + d, qa, qb, ones));
    acc2 = _mm256_add_epi32(acc2, dot_u8_i7_64(p2 + d, qa, qb, ones));
    acc3 = _mm256_add_epi32(acc3, dot_u8_i7_64(p3 + d, qa, qb, ones));
  }
  if (d + 32 <= stride_d) {
    __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d));
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + d)), qa), ones));
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + d)), qa), ones));
    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p2 + d)), qa), ones));
    acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p3 + d)), qa), ones));
  }
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out), hsum4_unpack(acc0, acc1, acc2, acc3));
}

inline bool prepare_u7_q(const QuantU7& u7, const float* q, int8_t* qi8,
                         double& base_const, float& two_s_qs, int qmax_i = 64) {
  const int D = u7.dim;
  float max_aq = 0.0f;
  double qnorm2 = 0.0, sum_minq = 0.0, sum_absq = 0.0;
  for (int d = 0; d < D; ++d) {
    float qd = q[d];
    if (!std::isfinite(qd)) return false;
    float ad = std::abs(qd);
    max_aq = std::max(max_aq, ad);
    qnorm2 += double(qd) * double(qd);
    sum_minq += double(u7.minv) * double(qd);
    sum_absq += double(ad);
  }
  if (max_aq > 1e6f || !std::isfinite(qnorm2)) return false;
  float qscale = (max_aq > 0.0f) ? (max_aq / float(qmax_i)) : 1.0f;
  float inv_qs = 1.0f / qscale;
  double sum_pos_r = 0.0, sum_rq = 0.0, rq2 = 0.0;
  for (int d = 0; d < D; ++d) {
    int qv = std::max(-qmax_i, std::min(qmax_i, int(std::lround(q[d] * inv_qs))));
    qi8[d] = int8_t(qv);
    double rq = double(q[d]) - double(qv) * double(qscale);
    if (rq > 0.0) sum_pos_r += rq;
    sum_rq += rq;
    rq2 += rq * rq;
  }
  for (int d = D; d < u7.stride; ++d) qi8[d] = 0;
  double deb_l1 = 2.0 * double(u7.scale) * 255.0 * sum_pos_r;
  double deb_cs = 2.0 * double(u7.scale) * 127.5 * sum_rq +
                  2.0 * double(u7.scale) * double(u7.max_udev) * std::sqrt(rq2);
  double deb = std::min(deb_l1, deb_cs);
  double f32 = 1e-5 * (u7.max_xhat_norm2 + qnorm2 +
                       2.0 * std::abs(double(u7.minv)) * sum_absq +
                       2.0 * double(u7.scale) * 255.0 * sum_absq + 1.0);
  base_const = qnorm2 - 2.0 * sum_minq - deb - f32;
  two_s_qs = 2.0f * u7.scale * qscale;
  return true;
}

inline void topk_push(std::vector<CacheCand>& top, size_t K, double d, int64_t pk, size_t slot) {
  if (top.size() < K) {
    top.push_back({d, pk, slot});
    size_t j = top.size() - 1;
    while (j > 0 && (top[j].d < top[j - 1].d || (top[j].d == top[j - 1].d && top[j].pk < top[j - 1].pk))) {
      std::swap(top[j], top[j - 1]); --j;
    }
  } else if (d < top.back().d || (d == top.back().d && pk < top.back().pk)) {
    top.back() = {d, pk, slot};
    size_t j = K - 1;
    while (j > 0 && (top[j].d < top[j - 1].d || (top[j].d == top[j - 1].d && top[j].pk < top[j - 1].pk))) {
      std::swap(top[j], top[j - 1]); --j;
    }
  }
}

inline void u7_exact_update_bound(const QuantU7& u7, const double base_const, const std::vector<CacheCand>& top,
                                  size_t K, float& tau_dist, float& tau_rhs) {
  if (top.size() == K) {
    tau_dist = float(top.back().d);
    double tb = double(tau_dist) + u7.max_row_err + 1e-4;
    double rhs = tb * tb - base_const;
    tau_rhs = float(rhs) + std::abs(float(rhs)) * 1e-6f + 1e-5f;
  }
}

// Keep the R smallest scores seen so far in `heap[0..nh)` sorted ascending;
// `tau` is the R-th smallest (the largest kept) once the buffer is full.
//
// This used to be a std::make_heap/pop_heap/push_heap max-heap. The heap
// helpers are ordinary (non-AVX) template code, so calling them from the
// AVX2 scan loop produced out-of-line calls into SSE code; a flat sorted
// buffer keeps the whole bookkeeping inline. Semantics are unchanged: a row
// is admitted iff its score is strictly below tau, and tau is the largest
// kept score.
//
// Measured on the 10k x 128 benchmark (interleaved A/B, 8 rounds, min):
// pass1 240.5 -> 245.4 ms (noise), pass2 43.4 -> 37.0 ms (-15%, candidates
// are now evaluated in score order so the bound tightens sooner), total
// flat. Kept because it is simpler and pass2 is faster; not claimed as a win.
inline void u7_topr_push(QuantCand* heap, size_t& nh, size_t R, float& tau, float sc, uint32_t s) {
  if (nh < R) {
    size_t i = nh++;
    while (i > 0 && heap[i - 1].sc > sc) {
      heap[i] = heap[i - 1];
      --i;
    }
    heap[i] = {sc, s};
    if (nh == R) tau = heap[R - 1].sc;
    return;
  }
  if (sc >= tau) return;
  size_t i = R - 1;
  while (i > 0 && heap[i - 1].sc > sc) {
    heap[i] = heap[i - 1];
    --i;
  }
  heap[i] = {sc, s};
  tau = heap[R - 1].sc;
}

// Evaluate top-R approximate candidates via exact L2, poison scs slots with NaN so
// ordered _CMP_LE_OQ skips them, then verify all remaining rows within the triangle bound.
__attribute__((target("avx2,fma")))
inline void u7_exact_pass2(const VecCache& c, const char* mbase, const int64_t* pks,
                           size_t begin, size_t len, const float* q, size_t K,
                           const QuantCand* heap, size_t nh, float* scs,
                           const double base_const, float& tau_dist, float& tau_rhs,
                           std::vector<CacheCand>& top) {
  const QuantU7& u7 = c.u7;
  const int dim = u7.dim;
  auto push = [&](double d, size_t s) {
    topk_push(top, K, d, pks[s], s);
    u7_exact_update_bound(u7, base_const, top, K, tau_dist, tau_rhs);
  };
  for (size_t k = 0; k < nh; ++k) {
    uint32_t s = heap[k].s;
    scs[s - begin] = std::numeric_limits<float>::quiet_NaN();
    push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim), s);
  }
  __m256 v_tau = _mm256_set1_ps(tau_rhs);
  size_t j = 0;
  for (; j + 8 <= len; j += 8) {
    __m256 v_sc = _mm256_loadu_ps(scs + j);
    int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_tau, _CMP_LE_OQ));
    if (__builtin_expect(mask != 0, 0)) {
      while (mask) {
        int b = __builtin_ctz(mask);
        mask &= mask - 1;
        size_t idx = j + size_t(b);
        size_t s = begin + idx;
        double rb = double(tau_dist) + u7.row_err[s] + 1e-4;
        if (double(scs[idx]) <= rb * rb - base_const) {
          push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim), s);
          v_tau = _mm256_set1_ps(tau_rhs);
        }
      }
    }
  }
  for (; j < len; ++j) {
    if (scs[j] <= tau_rhs) {
      size_t s = begin + j;
      double rb = double(tau_dist) + u7.row_err[s] + 1e-4;
      if (double(scs[j]) <= rb * rb - base_const)
        push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim), s);
    }
  }
}

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__) && (__GNUC__ >= 12)
__attribute__((target("avx2,fma,avxvnni")))
inline void u7_exact_scan_vnni_128(const VecCache& c, const char* mbase, const int64_t* pks,
                                   size_t begin, size_t len, const float* q, size_t K,
                                   std::vector<CacheCand>& top, std::vector<float>* scs_out) {
  const QuantU7& u7 = c.u7;
  const size_t end = begin + len;
  int8_t qi8[2048];
  double base_const = 0.0;
  float two_s_qs = 0.0f;
  // AVX-VNNI vpdpbusd has an int32 accumulator, avoiding int16 overflow at full [-127..127] range.
  if (!prepare_u7_q(u7, q, qi8, base_const, two_s_qs, 127)) {
    for (size_t s = begin; s < end; ++s) {
      double d = l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, u7.dim);
      topk_push(top, K, d, pks[s], s);
    }
    return;
  }
  const int dim = u7.dim;
  float tau_dist = 1e30f;
  float tau_rhs = 1e30f;
  auto push = [&](double d, size_t s) {
    topk_push(top, K, d, pks[s], s);
    u7_exact_update_bound(u7, base_const, top, K, tau_dist, tau_rhs);
  };
  const uint8_t* codes = u7.codes.data();
  const __m256 v_two = _mm256_set1_ps(two_s_qs);
  const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8));
  const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 32));
  const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 64));
  const __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 96));

  if (len >= 128 && K <= 64) {
    std::vector<float> local_scs;
    std::vector<float>& scs_vec = scs_out ? *scs_out : local_scs;
    if (scs_vec.size() < len) scs_vec.resize(len);
    float* scs = scs_vec.data();

    const size_t R = std::min<size_t>(len, std::min<size_t>(72, std::max<size_t>(K + 8, K * 2)));
    QuantCand heap[72];
    size_t nh = 0;
    float htau = 1e30f;
    __m256 v_htau = _mm256_set1_ps(htau);
    size_t i = begin;
    for (; i + 8 <= end; i += 8) {
      __m256i d8 = dot_u8_vnni_8x128(codes + i * 128, q0, q1, q2, q3);
      __m256 v_xn2 = _mm256_loadu_ps(u7.xhat_norm2.data() + i);
      __m256 v_sc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(d8), v_two, v_xn2);
      _mm256_storeu_ps(scs + (i - begin), v_sc);
      if (__builtin_expect(nh < R, 0)) {
        for (int b = 0; b < 8; ++b) u7_topr_push(heap, nh, R, htau, scs[i - begin + size_t(b)], uint32_t(i + size_t(b)));
        if (nh == R) v_htau = _mm256_set1_ps(htau);
      } else {
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_htau, _CMP_LT_OQ));
        if (__builtin_expect(mask != 0, 0)) {
          while (mask) {
            int b = __builtin_ctz(mask);
            mask &= mask - 1;
            float sv = scs[i - begin + size_t(b)];
            if (sv < htau) {
              u7_topr_push(heap, nh, R, htau, sv, uint32_t(i + size_t(b)));
              v_htau = _mm256_set1_ps(htau);
            }
          }
        }
      }
    }
    for (; i < end; ++i) {
      int32_t dot = 0;
      const uint8_t* pc = codes + i * 128;
      for (int d = 0; d < dim; ++d) dot += int32_t(pc[d]) * int32_t(qi8[d]);
      float sc = u7.xhat_norm2[i] - two_s_qs * float(dot);
      scs[i - begin] = sc;
      if (nh < R || sc < htau) u7_topr_push(heap, nh, R, htau, sc, uint32_t(i));
    }
    u7_exact_pass2(c, mbase, pks, begin, len, q, K, heap, nh, scs, base_const, tau_dist, tau_rhs, top);
    return;
  }

  size_t seed = std::min<size_t>(end, begin + K);
  size_t i = begin;
  for (; i < seed; ++i) push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[i]), q, dim), i);
  __m256 v_tau = _mm256_set1_ps(tau_rhs);
  for (; i + 8 <= end; i += 8) {
    __m256i d8 = dot_u8_vnni_8x128(codes + i * 128, q0, q1, q2, q3);
    __m256 v_xn2 = _mm256_loadu_ps(u7.xhat_norm2.data() + i);
    __m256 v_sc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(d8), v_two, v_xn2);
    int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_tau, _CMP_LE_OQ));
    if (__builtin_expect(mask != 0, 0)) {
      alignas(32) float scs[8];
      _mm256_store_ps(scs, v_sc);
      while (mask) {
        int b = __builtin_ctz(mask);
        mask &= mask - 1;
        size_t s = i + size_t(b);
        double rb = double(tau_dist) + u7.row_err[s] + 1e-4;
        if (double(scs[b]) <= rb * rb - base_const) {
          push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim), s);
          v_tau = _mm256_set1_ps(tau_rhs);
        }
      }
    }
  }
  for (; i < end; ++i) push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[i]), q, dim), i);
}
#endif

__attribute__((target("avx2,fma")))
inline void u7_exact_scan_contig_range(const VecCache& c, const char* mbase, const int64_t* pks,
                                        size_t begin, size_t len, const float* q, size_t K, std::vector<CacheCand>& top,
                                        std::vector<float>* scs_out = nullptr) {
  if (K == 0 || len == 0) return;
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__) && (__GNUC__ >= 12)
  if (g_has_avxvnni && c.u7.stride == 128 && len >= 128) {
    u7_exact_scan_vnni_128(c, mbase, pks, begin, len, q, K, top, scs_out);
    return;
  }
#endif
  const QuantU7& u7 = c.u7;
  const size_t end = begin + len;
  int8_t qi8[2048];
  double base_const = 0.0;
  float two_s_qs = 0.0f;
  if (!prepare_u7_q(u7, q, qi8, base_const, two_s_qs)) {
    for (size_t s = begin; s < end; ++s) {
      double d = l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, u7.dim);
      topk_push(top, K, d, pks[s], s);
    }
    return;
  }
  const int dim = u7.dim;
  const int stride = u7.stride;
  float tau_dist = 1e30f;
  float tau_rhs = 1e30f;
  auto push = [&](double d, size_t s) {
    topk_push(top, K, d, pks[s], s);
    u7_exact_update_bound(u7, base_const, top, K, tau_dist, tau_rhs);
  };
  const uint8_t* codes = u7.codes.data();
  __m256 v_two = _mm256_set1_ps(two_s_qs);

  // Two-pass exact filter for moderate K: first find tight true-NN tau via top-R approx
  // scores so Pass 2 only invokes l2_kernel on ~15 rows instead of ~100+.
  if (len >= 128 && K <= 64) {
    std::vector<float> local_scs;
    std::vector<float>& scs_vec = scs_out ? *scs_out : local_scs;
    if (scs_vec.size() < len) scs_vec.resize(len);
    float* scs = scs_vec.data();

    const size_t R = std::min<size_t>(len, std::min<size_t>(72, std::max<size_t>(K + 8, K * 2)));
    QuantCand heap[72];
    size_t nh = 0;
    float htau = 1e30f;
    __m256 v_htau = _mm256_set1_ps(htau);
    size_t i = begin;
    if (stride == 128) {
      const __m256i ones = _mm256_set1_epi16(1);
      const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8));
      const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 32));
      const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 64));
      const __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 96));
      for (; i + 8 <= end; i += 8) {
        __m256i d8 = dot_u8_i7_8x128(codes + i * 128, q0, q1, q2, q3, ones);
        __m256 v_xn2 = _mm256_loadu_ps(u7.xhat_norm2.data() + i);
        __m256 v_sc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(d8), v_two, v_xn2);
        _mm256_storeu_ps(scs + (i - begin), v_sc);
        if (__builtin_expect(nh < R, 0)) {
          for (int b = 0; b < 8; ++b) u7_topr_push(heap, nh, R, htau, scs[i - begin + size_t(b)], uint32_t(i + size_t(b)));
          if (nh == R) v_htau = _mm256_set1_ps(htau);
        } else {
          int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_htau, _CMP_LT_OQ));
          if (__builtin_expect(mask != 0, 0)) {
            while (mask) {
              int b = __builtin_ctz(mask);
              mask &= mask - 1;
              float sv = scs[i - begin + size_t(b)];
              if (sv < htau) {
                u7_topr_push(heap, nh, R, htau, sv, uint32_t(i + size_t(b)));
                v_htau = _mm256_set1_ps(htau);
              }
            }
          }
        }
      }
    } else {
      for (; i + 8 <= end; i += 8) {
        int32_t dots[8];
        dot_u7_i6_4xD(codes + i * size_t(stride), size_t(stride), qi8, stride, dots);
        dot_u7_i6_4xD(codes + (i + 4) * size_t(stride), size_t(stride), qi8, stride, dots + 4);
        __m256 v_xn2 = _mm256_loadu_ps(u7.xhat_norm2.data() + i);
        __m256 v_dot = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dots)));
        __m256 v_sc = _mm256_fnmadd_ps(v_dot, v_two, v_xn2);
        _mm256_storeu_ps(scs + (i - begin), v_sc);
        if (__builtin_expect(nh < R, 0)) {
          for (int b = 0; b < 8; ++b) u7_topr_push(heap, nh, R, htau, scs[i - begin + size_t(b)], uint32_t(i + size_t(b)));
          if (nh == R) v_htau = _mm256_set1_ps(htau);
        } else {
          int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_htau, _CMP_LT_OQ));
          if (__builtin_expect(mask != 0, 0)) {
            while (mask) {
              int b = __builtin_ctz(mask);
              mask &= mask - 1;
              float sv = scs[i - begin + size_t(b)];
              if (sv < htau) {
                u7_topr_push(heap, nh, R, htau, sv, uint32_t(i + size_t(b)));
                v_htau = _mm256_set1_ps(htau);
              }
            }
          }
        }
      }
    }
    for (; i < end; ++i) {
      int32_t dot = 0;
      const uint8_t* pc = codes + i * size_t(stride);
      for (int d = 0; d < dim; ++d) dot += int32_t(pc[d]) * int32_t(qi8[d]);
      float sc = u7.xhat_norm2[i] - two_s_qs * float(dot);
      scs[i - begin] = sc;
      if (nh < R || sc < htau) u7_topr_push(heap, nh, R, htau, sc, uint32_t(i));
    }

    u7_exact_pass2(c, mbase, pks, begin, len, q, K, heap, nh, scs, base_const, tau_dist, tau_rhs, top);
    return;
  }

  size_t seed = std::min<size_t>(end, begin + K);
  size_t i = begin;
  for (; i < seed; ++i) push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[i]), q, dim), i);

  __m256 v_tau = _mm256_set1_ps(tau_rhs);
  if (stride == 128) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8));
    const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 32));
    const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 64));
    const __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 96));
    for (; i + 8 <= end; i += 8) {
      __m256i d8 = dot_u8_i7_8x128(codes + i * 128, q0, q1, q2, q3, ones);
      __m256 v_xn2 = _mm256_loadu_ps(u7.xhat_norm2.data() + i);
      __m256 v_sc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(d8), v_two, v_xn2);
      int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_tau, _CMP_LE_OQ));
      if (__builtin_expect(mask != 0, 0)) {
        alignas(32) float scs[8];
        _mm256_store_ps(scs, v_sc);
        while (mask) {
          int b = __builtin_ctz(mask);
          mask &= mask - 1;
          size_t s = i + size_t(b);
          double rb = double(tau_dist) + u7.row_err[s] + 1e-4;
          if (double(scs[b]) <= rb * rb - base_const) {
            push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim), s);
            v_tau = _mm256_set1_ps(tau_rhs);
          }
        }
      }
    }
  } else {
    for (; i + 8 <= end; i += 8) {
      int32_t dots[8];
      dot_u7_i6_4xD(codes + i * size_t(stride), size_t(stride), qi8, stride, dots);
      dot_u7_i6_4xD(codes + (i + 4) * size_t(stride), size_t(stride), qi8, stride, dots + 4);
      __m256 v_xn2 = _mm256_loadu_ps(u7.xhat_norm2.data() + i);
      __m256 v_dot = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dots)));
      __m256 v_sc = _mm256_fnmadd_ps(v_dot, v_two, v_xn2);
      int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_tau, _CMP_LE_OQ));
      if (__builtin_expect(mask != 0, 0)) {
        alignas(32) float scs[8];
        _mm256_store_ps(scs, v_sc);
        while (mask) {
          int b = __builtin_ctz(mask);
          mask &= mask - 1;
          size_t s = i + size_t(b);
          double rb = double(tau_dist) + u7.row_err[s] + 1e-4;
          if (double(scs[b]) <= rb * rb - base_const) {
            push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim), s);
            v_tau = _mm256_set1_ps(tau_rhs);
          }
        }
      }
    }
  }
  for (; i < end; ++i) push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[i]), q, dim), i);
}

__attribute__((target("avx2,fma")))
inline void u7_exact_scan_indirect(const VecCache& c, const char* mbase, const int64_t* pks,
                                    const uint32_t* ids, size_t M, const float* q, size_t K, std::vector<CacheCand>& top) {
  if (K == 0 || M == 0) return;
  const QuantU7& u7 = c.u7;
  int8_t qi8[2048];
  double base_const = 0.0;
  float two_s_qs = 0.0f;
  if (!prepare_u7_q(u7, q, qi8, base_const, two_s_qs)) {
    for (size_t i = 0; i < M; ++i) {
      uint32_t s = ids[i];
      double d = l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, u7.dim);
      topk_push(top, K, d, pks[s], s);
    }
    return;
  }
  const int dim = u7.dim;
  const int stride = u7.stride;
  float tau_dist = 1e30f;
  float tau_rhs = 1e30f;
  auto push = [&](double d, uint32_t s) {
    topk_push(top, K, d, pks[s], s);
    if (top.size() == K) {
      tau_dist = float(top.back().d);
      double tb = double(tau_dist) + u7.max_row_err + 1e-4;
      double rhs = tb * tb - base_const;
      tau_rhs = float(rhs) + std::abs(float(rhs)) * 1e-6f + 1e-5f;
    }
  };
  size_t seed = std::min<size_t>(M, std::max<size_t>(K, 24));
  size_t i = 0;
  for (; i < seed; ++i) {
    uint32_t s = ids[i];
    push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim), s);
  }
  __m128 v_two = _mm_set1_ps(two_s_qs);
  __m128 v_tau = _mm_set1_ps(tau_rhs);
  const uint8_t* codes = u7.codes.data();
  for (; i + 4 <= M; i += 4) {
    int32_t dots[4];
    dot_u7_i6_4x_indirect(codes, ids + i, size_t(stride), qi8, stride, dots);
    __m128 v_xn2 = _mm_set_ps(u7.xhat_norm2[ids[i + 3]], u7.xhat_norm2[ids[i + 2]],
                              u7.xhat_norm2[ids[i + 1]], u7.xhat_norm2[ids[i]]);
    __m128 v_dot = _mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dots)));
    __m128 v_sc = _mm_fnmadd_ps(v_dot, v_two, v_xn2);
    int mask = _mm_movemask_ps(_mm_cmple_ps(v_sc, v_tau));
    if (__builtin_expect(mask != 0, 0)) {
      alignas(16) float scs[4];
      _mm_store_ps(scs, v_sc);
      while (mask) {
        int b = __builtin_ctz(mask);
        mask &= mask - 1;
        uint32_t s = ids[i + size_t(b)];
        double rb = double(tau_dist) + u7.row_err[s] + 1e-4;
        if (double(scs[b]) <= rb * rb - base_const) {
          push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim), s);
          v_tau = _mm_set1_ps(tau_rhs);
        }
      }
    }
  }
  for (; i < M; ++i) {
    uint32_t s = ids[i];
    push(l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim), s);
  }
}

inline bool u7_ann_prepare_q(const QuantU7& u7, const float* q, int qmax_i, int8_t* qi8, float& two_s_qs) {
  const int dim = u7.dim;
  const int stride = u7.stride;
  float max_aq = 0.0f;
  for (int d = 0; d < dim; ++d) {
    if (!std::isfinite(q[d])) { max_aq = 1e30f; break; }
    max_aq = std::max(max_aq, std::abs(q[d]));
  }
  if (!u7.allow_ann || max_aq > 1e6f) return false;
  float qscale = (max_aq > 0.0f) ? (max_aq / float(qmax_i)) : 1.0f;
  float inv_qs = 1.0f / qscale;
  for (int d = 0; d < dim; ++d)
    qi8[d] = int8_t(std::max(-qmax_i, std::min(qmax_i, int(std::lround(q[d] * inv_qs)))));
  for (int d = dim; d < stride; ++d) qi8[d] = 0;
  two_s_qs = 2.0f * u7.scale * qscale;
  return true;
}

inline void u7_ann_rerank_eval(const VecCache& c, const char* mbase, const int64_t* pks,
                               const float* q, size_t K, const QuantCand* heap, size_t nh,
                               std::vector<CacheCand>& top) {
  const int dim = c.u7.dim;
  for (size_t j = 0; j < nh; ++j) {
    uint32_t s = heap[j].s;
    double d = l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), q, dim);
    topk_push(top, K, d, pks[s], s);
  }
}

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__) && (__GNUC__ >= 12)
__attribute__((target("avx2,fma,avxvnni")))
inline void u7_ann_rerank_vnni_128(const VecCache& c, const char* mbase, const int64_t* pks,
                                   size_t begin, size_t len, const float* q, size_t K,
                                   std::vector<CacheCand>& top, std::vector<float>* scs_out) {
  const QuantU7& u7 = c.u7;
  const int dim = u7.dim;
  const size_t end = begin + len;
  int8_t qi8[2048];
  float two_s_qs = 0.0f;
  if (!u7_ann_prepare_q(u7, q, 127, qi8, two_s_qs)) {
    u7_exact_scan_vnni_128(c, mbase, pks, begin, len, q, K, top, scs_out);
    return;
  }
  const size_t R = std::min<size_t>(len, std::min<size_t>(140, std::max<size_t>(K + 6, (K * 3 + 1) / 2)));
  QuantCand heap[140];
  size_t nh = 0;
  float tau = 1e30f;
  const uint8_t* codes = u7.codes.data();
  const __m256 v_two = _mm256_set1_ps(two_s_qs);
  __m256 v_tau = _mm256_set1_ps(tau);
  const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8));
  const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 32));
  const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 64));
  const __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 96));
  size_t i = begin;
  for (; i + 8 <= end; i += 8) {
    __m256i d8 = dot_u8_vnni_8x128(codes + i * 128, q0, q1, q2, q3);
    __m256 v_xn2 = _mm256_loadu_ps(u7.xhat_norm2.data() + i);
    __m256 v_sc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(d8), v_two, v_xn2);
    if (__builtin_expect(nh < R, 0)) {
      alignas(32) float scs[8];
      _mm256_store_ps(scs, v_sc);
      for (int b = 0; b < 8; ++b) u7_topr_push(heap, nh, R, tau, scs[b], uint32_t(i + size_t(b)));
      if (nh == R) v_tau = _mm256_set1_ps(tau);
    } else {
      int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_tau, _CMP_LT_OQ));
      if (__builtin_expect(mask != 0, 0)) {
        alignas(32) float scs[8];
        _mm256_store_ps(scs, v_sc);
        while (mask) {
          int b = __builtin_ctz(mask);
          mask &= mask - 1;
          if (scs[b] < tau) {
            u7_topr_push(heap, nh, R, tau, scs[b], uint32_t(i + size_t(b)));
            v_tau = _mm256_set1_ps(tau);
          }
        }
      }
    }
  }
  for (; i < end; ++i) {
    int32_t dot = 0;
    const uint8_t* pc = codes + i * 128;
    for (int d = 0; d < dim; ++d) dot += int32_t(pc[d]) * int32_t(qi8[d]);
    float sc = u7.xhat_norm2[i] - two_s_qs * float(dot);
    if (nh < R || sc < tau) u7_topr_push(heap, nh, R, tau, sc, uint32_t(i));
  }
  u7_ann_rerank_eval(c, mbase, pks, q, K, heap, nh, top);
}
#endif

__attribute__((target("avx2,fma")))
inline void u7_ann_rerank_contig_range(const VecCache& c, const char* mbase, const int64_t* pks,
                                        size_t begin, size_t len, const float* q, size_t K, std::vector<CacheCand>& top,
                                        std::vector<float>* scs_out = nullptr) {
  if (K == 0 || len == 0) return;
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__) && (__GNUC__ >= 12)
  if (g_has_avxvnni && c.u7.stride == 128 && len >= 128) {
    u7_ann_rerank_vnni_128(c, mbase, pks, begin, len, q, K, top, scs_out);
    return;
  }
#endif
  const QuantU7& u7 = c.u7;
  const int dim = u7.dim;
  const int stride = u7.stride;
  const size_t end = begin + len;
  int8_t qi8[2048];
  float two_s_qs = 0.0f;
  if (!u7_ann_prepare_q(u7, q, 64, qi8, two_s_qs)) {
    u7_exact_scan_contig_range(c, mbase, pks, begin, len, q, K, top, scs_out);
    return;
  }

  const size_t R = std::min<size_t>(len, std::min<size_t>(140, std::max<size_t>(K + 6, (K * 3 + 1) / 2)));
  QuantCand heap[140];
  size_t nh = 0;
  float tau = 1e30f;
  const uint8_t* codes = u7.codes.data();
  __m256 v_two = _mm256_set1_ps(two_s_qs);
  __m256 v_tau = _mm256_set1_ps(tau);
  size_t i = begin;
  if (stride == 128) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8));
    const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 32));
    const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 64));
    const __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qi8 + 96));
    for (; i + 8 <= end; i += 8) {
      __m256i d8 = dot_u8_i7_8x128(codes + i * 128, q0, q1, q2, q3, ones);
      __m256 v_xn2 = _mm256_loadu_ps(u7.xhat_norm2.data() + i);
      __m256 v_sc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(d8), v_two, v_xn2);
      if (__builtin_expect(nh < R, 0)) {
        alignas(32) float scs[8];
        _mm256_store_ps(scs, v_sc);
        for (int b = 0; b < 8; ++b) u7_topr_push(heap, nh, R, tau, scs[b], uint32_t(i + size_t(b)));
        if (nh == R) v_tau = _mm256_set1_ps(tau);
      } else {
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_tau, _CMP_LT_OQ));
        if (__builtin_expect(mask != 0, 0)) {
          alignas(32) float scs[8];
          _mm256_store_ps(scs, v_sc);
          while (mask) {
            int b = __builtin_ctz(mask);
            mask &= mask - 1;
            if (scs[b] < tau) {
              u7_topr_push(heap, nh, R, tau, scs[b], uint32_t(i + size_t(b)));
              v_tau = _mm256_set1_ps(tau);
            }
          }
        }
      }
    }
  } else {
    for (; i + 8 <= end; i += 8) {
      int32_t dots[8];
      dot_u7_i6_4xD(codes + i * size_t(stride), size_t(stride), qi8, stride, dots);
      dot_u7_i6_4xD(codes + (i + 4) * size_t(stride), size_t(stride), qi8, stride, dots + 4);
      __m256 v_xn2 = _mm256_loadu_ps(u7.xhat_norm2.data() + i);
      __m256 v_dot = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dots)));
      __m256 v_sc = _mm256_fnmadd_ps(v_dot, v_two, v_xn2);
      if (__builtin_expect(nh < R, 0)) {
        alignas(32) float scs[8];
        _mm256_store_ps(scs, v_sc);
        for (int b = 0; b < 8; ++b) u7_topr_push(heap, nh, R, tau, scs[b], uint32_t(i + size_t(b)));
        if (nh == R) v_tau = _mm256_set1_ps(tau);
      } else {
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_sc, v_tau, _CMP_LT_OQ));
        if (__builtin_expect(mask != 0, 0)) {
          alignas(32) float scs[8];
          _mm256_store_ps(scs, v_sc);
          while (mask) {
            int b = __builtin_ctz(mask);
            mask &= mask - 1;
            if (scs[b] < tau) {
              u7_topr_push(heap, nh, R, tau, scs[b], uint32_t(i + size_t(b)));
              v_tau = _mm256_set1_ps(tau);
            }
          }
        }
      }
    }
  }
  for (; i < end; ++i) {
    int32_t dot = 0;
    const uint8_t* pc = codes + i * size_t(stride);
    for (int d = 0; d < dim; ++d) dot += int32_t(pc[d]) * int32_t(qi8[d]);
    float sc = u7.xhat_norm2[i] - two_s_qs * float(dot);
    if (nh < R || sc < tau) u7_topr_push(heap, nh, R, tau, sc, uint32_t(i));
  }
  u7_ann_rerank_eval(c, mbase, pks, q, K, heap, nh, top);
}

__attribute__((target("avx2,fma")))
inline void u7_ann_rerank_indirect(const VecCache& c, const char* mbase, const int64_t* pks,
                                    const uint32_t* ids, size_t M, const float* q, size_t K, std::vector<CacheCand>& top) {
  if (K == 0 || M == 0) return;
  const QuantU7& u7 = c.u7;
  const int dim = u7.dim;
  const int stride = u7.stride;
  int8_t qi8[2048];
  float two_s_qs = 0.0f;
  if (!u7_ann_prepare_q(u7, q, 64, qi8, two_s_qs)) {
    u7_exact_scan_indirect(c, mbase, pks, ids, M, q, K, top);
    return;
  }

  const size_t R = std::min<size_t>(M, std::min<size_t>(140, std::max<size_t>(K + 6, (K * 3 + 1) / 2)));
  QuantCand heap[140];
  size_t nh = 0;
  float tau = 1e30f;
  const uint8_t* codes = u7.codes.data();
  __m128 v_two = _mm_set1_ps(two_s_qs);
  __m128 v_tau = _mm_set1_ps(tau);
  size_t i = 0;
  for (; i + 4 <= M; i += 4) {
    int32_t dots[4];
    dot_u7_i6_4x_indirect(codes, ids + i, size_t(stride), qi8, stride, dots);
    __m128 v_xn2 = _mm_set_ps(u7.xhat_norm2[ids[i + 3]], u7.xhat_norm2[ids[i + 2]],
                              u7.xhat_norm2[ids[i + 1]], u7.xhat_norm2[ids[i]]);
    __m128 v_dot = _mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dots)));
    __m128 v_sc = _mm_fnmadd_ps(v_dot, v_two, v_xn2);
    if (__builtin_expect(nh < R, 0)) {
      alignas(16) float scs[4];
      _mm_store_ps(scs, v_sc);
      for (int b = 0; b < 4; ++b) u7_topr_push(heap, nh, R, tau, scs[b], ids[i + size_t(b)]);
      if (nh == R) v_tau = _mm_set1_ps(tau);
    } else {
      int mask = _mm_movemask_ps(_mm_cmple_ps(v_sc, v_tau));
      if (__builtin_expect(mask != 0, 0)) {
        alignas(16) float scs[4];
        _mm_store_ps(scs, v_sc);
        while (mask) {
          int b = __builtin_ctz(mask);
          mask &= mask - 1;
          if (scs[b] < tau) {
            u7_topr_push(heap, nh, R, tau, scs[b], ids[i + size_t(b)]);
            v_tau = _mm_set1_ps(tau);
          }
        }
      }
    }
  }
  for (; i < M; ++i) {
    uint32_t s = ids[i];
    int32_t dot = 0;
    const uint8_t* pc = codes + size_t(s) * size_t(stride);
    for (int d = 0; d < dim; ++d) dot += int32_t(pc[d]) * int32_t(qi8[d]);
    float sc = u7.xhat_norm2[s] - two_s_qs * float(dot);
    if (nh < R || sc < tau) u7_topr_push(heap, nh, R, tau, sc, s);
  }
  u7_ann_rerank_eval(c, mbase, pks, q, K, heap, nh, top);
}
#endif

struct PkIndex {
  struct Entry {
    int64_t pk;
    size_t off;
  };
  static constexpr size_t kTomb = ~size_t(0);

  std::vector<Entry> base_;
  std::map<int64_t, size_t> delta_;
  size_t live_ = 0;

  bool empty() const { return live_ == 0; }
  size_t size() const { return live_; }
  size_t base_size() const { return base_.size(); }
  bool delta_empty() const { return delta_.empty(); }

  void clear() {
    base_.clear();
    delta_.clear();
    live_ = 0;
  }

  void reserve(size_t n) {
    if (delta_.empty() && base_.empty()) base_.reserve(n);
  }

  size_t base_find(int64_t pk) const {
    auto it = std::lower_bound(base_.begin(), base_.end(), pk,
                               [](const Entry& e, int64_t k) { return e.pk < k; });
    if (it != base_.end() && it->pk == pk) return size_t(it - base_.begin());
    return kTomb;
  }

  bool find(int64_t pk, size_t& out) const {
    if (!delta_.empty()) {
      auto dit = delta_.find(pk);
      if (dit != delta_.end()) {
        if (dit->second == kTomb) return false;
        out = dit->second;
        return true;
      }
    }
    size_t bi = base_find(pk);
    if (bi != kTomb) {
      out = base_[bi].off;
      return true;
    }
    return false;
  }

  bool contains(int64_t pk) const {
    size_t dummy = 0;
    return find(pk, dummy);
  }

  size_t at(int64_t pk) const {
    size_t v = 0;
    find(pk, v);
    return v;
  }

  bool insert_unique(int64_t pk, size_t off) {
    if (delta_.empty() && (base_.empty() || pk > base_.back().pk)) {
      base_.push_back({pk, off});
      ++live_;
      return true;
    }
    if (!delta_.empty()) {
      auto dit = delta_.find(pk);
      if (dit != delta_.end()) {
        if (dit->second != kTomb) return false;
        dit->second = off;
        ++live_;
        return true;
      }
    }
    if (base_find(pk) != kTomb) return false;
    delta_.emplace(pk, off);
    ++live_;
    maybe_compact();
    return true;
  }

  void set(int64_t pk, size_t off) {
    if (delta_.empty() && (base_.empty() || pk > base_.back().pk)) {
      base_.push_back({pk, off});
      ++live_;
      return;
    }
    if (!delta_.empty()) {
      auto dit = delta_.find(pk);
      if (dit != delta_.end()) {
        if (dit->second == kTomb) ++live_;
        dit->second = off;
        return;
      }
    }
    size_t bi = base_find(pk);
    if (bi != kTomb) {
      base_[bi].off = off;
      return;
    }
    delta_.emplace(pk, off);
    ++live_;
    maybe_compact();
  }

  void erase(int64_t pk) {
    if (!base_.empty() && base_.back().pk == pk &&
        (delta_.empty() || delta_.find(pk) == delta_.end())) {
      base_.pop_back();
      --live_;
      return;
    }
    if (!delta_.empty()) {
      auto dit = delta_.find(pk);
      if (dit != delta_.end()) {
        if (dit->second == kTomb) return;
        --live_;
        if (base_find(pk) != kTomb) dit->second = kTomb;
        else delta_.erase(dit);
        return;
      }
    }
    if (base_find(pk) != kTomb) {
      delta_.emplace(pk, kTomb);
      --live_;
      maybe_compact();
    }
  }

  void patch_tail(size_t base_start, size_t count, size_t pay_base, size_t rlen) {
    for (size_t k = 0; k < count; ++k) base_[base_start + k].off = pay_base + k * rlen;
  }

  void compact() {
    if (delta_.empty()) return;
    std::vector<Entry> out;
    out.reserve(live_);
    size_t bi = 0;
    const size_t bn = base_.size();
    auto dit = delta_.begin();
    const auto dend = delta_.end();
    while (bi < bn && dit != dend) {
      if (base_[bi].pk < dit->first) {
        out.push_back(base_[bi++]);
      } else if (dit->first < base_[bi].pk) {
        if (dit->second != kTomb) out.push_back({dit->first, dit->second});
        ++dit;
      } else {
        if (dit->second != kTomb) out.push_back({dit->first, dit->second});
        ++bi;
        ++dit;
      }
    }
    while (bi < bn) out.push_back(base_[bi++]);
    while (dit != dend) {
      if (dit->second != kTomb) out.push_back({dit->first, dit->second});
      ++dit;
    }
    base_.swap(out);
    delta_.clear();
  }

  void maybe_compact() {
    if (delta_.size() > std::max<size_t>(256, base_.size() / 4)) compact();
  }

  void shrink_excess() {
    if (base_.capacity() > base_.size() + base_.size() / 4) std::vector<Entry>(base_).swap(base_);
  }

  template <class Fn>
  void for_each_while(Fn&& fn) const {
    if (delta_.empty()) {
      for (const Entry& e : base_) {
        if (!fn(e.pk, e.off)) break;
      }
      return;
    }
    size_t bi = 0;
    const size_t bn = base_.size();
    auto dit = delta_.begin();
    const auto dend = delta_.end();
    while (bi < bn && dit != dend) {
      if (base_[bi].pk < dit->first) {
        if (!fn(base_[bi].pk, base_[bi].off)) return;
        ++bi;
      } else if (dit->first < base_[bi].pk) {
        if (dit->second != kTomb && !fn(dit->first, dit->second)) return;
        ++dit;
      } else {
        if (dit->second != kTomb && !fn(dit->first, dit->second)) return;
        ++bi;
        ++dit;
      }
    }
    while (bi < bn) {
      if (!fn(base_[bi].pk, base_[bi].off)) return;
      ++bi;
    }
    while (dit != dend) {
      if (dit->second != kTomb && !fn(dit->first, dit->second)) return;
      ++dit;
    }
  }
};

struct Table {
  std::string name;
  std::vector<sql::ColumnDef> cols;
  int pk_index = 0;
  int vec_index = -1;
  bool has_vector_index = false;
  std::string index_name;
  size_t fixed_rlen = 0;  // >0 when no VarChar: index & cache.offs store payload offset directly

  File rec;
  int file_id = 0;
  PkIndex index;  // pk -> record (or payload when fixed_rlen>0) offset
  VecCache cache;
  // Read-only mmap of rec, created lazily by ensure_cache() only for tables
  // that fit the memory budget (a large table must not reserve address space).
  char* map_base = nullptr;
  size_t map_len = 0;

  ~Table() { if (map_base) ::munmap(map_base, map_len); }

  int col_pos(const std::string& n) const {
    for (size_t k = 0; k < cols.size(); ++k) if (cols[k].name == n) return int(k);
    return -1;
  }
};

struct UndoEntry {
  int file_id;
  int64_t pk;
  bool had;
  size_t off;
};

// ------------------------------------------------------------------ engine

class Engine {
 public:
  explicit Engine(std::string dir) : dir_(std::move(dir)) { dbprobe::calibrate(); recover(); }

  mj::Ptr execute(const std::string& text, const std::string& mode) {
    sql::Statement st;
    try {
      st = sql::parse(text);
    } catch (const sql::Error& e) {
      return error_result(e.code, e.what());
    }
    search_mode_ = mode;
    l2_cache_.clear();
    size_t mark = undo_.size();
    bool autocommit = !in_txn_ && needs_txn(st.kind);
    if (autocommit) begin_txn();
    try {
      mj::Ptr out = run(st);
      if (autocommit) commit();
      return out;
    } catch (const Error& e) {
      rollback_to(mark, !autocommit);
      if (autocommit) abort_txn();
      return error_result(e.code, e.what());
    } catch (const sql::Error& e) {
      rollback_to(mark, !autocommit);
      if (autocommit) abort_txn();
      return error_result(e.code, e.what());
    }
  }

  void execute_jsonl(const mj::Ptr& id, const std::string& text, const std::string& mode, std::string& out_line) {
    fast_id_buf_.clear();
    mj::dump(fast_id_buf_, id);
    execute_jsonl_sv(fast_id_buf_, text, mode, out_line);
  }

  // Zero-alloc JSONL fast path for batch INSERT and TopK SELECT; falls back
  // transparently to execute() + mj::dump on any unusual syntax or error.
  void execute_jsonl_sv(std::string_view id_raw, std::string_view text, std::string_view mode, std::string& out_line) {
    out_line.clear();
    if (try_fast_select_topk(id_raw, text, mode, out_line)) return;
    if (try_fast_insert(id_raw, text, out_line)) return;
    fast_sql_buf_.assign(text.data(), text.size());
    fast_mode_buf_.assign(mode.data(), mode.size());
    mj::Ptr body = execute(fast_sql_buf_, fast_mode_buf_);
    if (fast_sql_buf_.capacity() > (128u << 10)) std::string().swap(fast_sql_buf_);
    out_line += "{\"id\":";
    out_line.append(id_raw.data(), id_raw.size());
    for (const auto& kv : body->obj) {
      out_line += ',';
      mj::dump_string(out_line, kv.first);
      out_line += ':';
      mj::dump(out_line, kv.second);
    }
    out_line += "}\n";
  }

  void shutdown() {
    if (in_txn_) { rollback_to(0); abort_txn(); }
    flush_wbuf();
    for (auto& t : tables_) t.second->rec.sync();
  }

 private:
  std::string dir_;
  std::map<std::string, std::shared_ptr<Table>, std::less<>> tables_;
  std::vector<std::shared_ptr<Table>> by_id_;

  Table* find_table(std::string_view name) const {
    if (by_id_.size() == 1) {
      const auto& t = by_id_[0];
      return (t && t->name == name) ? t.get() : nullptr;
    }
    auto it = tables_.find(name);
    return (it == tables_.end()) ? nullptr : it->second.get();
  }
  File commits_;
  std::set<uint64_t> committed_;
  uint64_t next_txn_ = 1;
  uint64_t cur_txn_ = 0;
  bool in_txn_ = false;
  std::vector<UndoEntry> undo_;
  std::set<int> dirty_;
  uint64_t max_txn_seen_ = 0;  // includes aborted transactions: ids must never be reused
  struct PredKey { int col; sql::CmpOp op; int64_t iv; };
  std::string search_mode_;
  std::unordered_map<const void*, std::vector<float>> l2_cache_;  // per statement
  std::vector<PredKey> fast_preds_;
  std::vector<float> fast_qvec_;
  std::vector<uint32_t> fast_slots_;
  std::vector<CacheCand> fast_top_;
  std::vector<float> fast_scs_;
  std::string fast_id_buf_;
  std::string fast_sql_buf_;
  std::string fast_mode_buf_;
  Row scratch_;                 // reused decode target: no per-row allocation
  std::vector<char> scratch_buf_;
  BlockCache cache_;
  Row ins_row_;                 // reused INSERT row: vector capacity survives
  std::string payload_buf_;     // encode_row output, reused across rows
  std::string entry_buf_;       // one framed record, reused across rows
  // Statement-level write batching: entries accumulate here and land in the
  // record file with ONE pwrite at the next statement boundary / commit /
  // rollback / shutdown. Committed bytes always reach the page cache before
  // the response is emitted, so the process-kill durability model holds.
  std::string wbuf_;
  Table* wbuf_tb_ = nullptr;
  size_t wbuf_off_ = 0;
  std::string batch_buf_;
  std::vector<int64_t> batch_pks_;

  static bool needs_txn(sql::StmtKind k) {
    return k == sql::StmtKind::Insert || k == sql::StmtKind::Update || k == sql::StmtKind::Delete ||
           k == sql::StmtKind::CreateTable || k == sql::StmtKind::CreateVectorIndex;
  }

  static mj::Ptr error_result(const std::string& code, const std::string& msg) {
    auto r = mj::Value::object();
    r->set("ok", mj::Value::boolean(false));
    auto e = mj::Value::object();
    e->set("code", mj::Value::str(code.empty() ? "internal_error" : code));
    e->set("message", mj::Value::str(msg));
    r->set("error", e);
    return r;
  }
  static mj::Ptr ok_result() {
    auto r = mj::Value::object();
    r->set("ok", mj::Value::boolean(true));
    return r;
  }
  static mj::Ptr affected(int64_t n) {
    auto r = ok_result();
    r->set("affected_rows", mj::Value::integer(n));
    return r;
  }

  std::string path(const std::string& name) const { return dir_ + "/" + name; }

  // ---------------------------------------------------------- recovery

  void recover() {
    ::mkdir(dir_.c_str(), 0755);
    commits_.open(path("commits.log"));
    {
      size_t off = 0;
      char buf[12];
      while (off + 12 <= commits_.size) {
        commits_.read_at(off, buf, 12);
        uint64_t txn = get_u64(buf);
        uint32_t crc = get_u32(buf + 8);
        if (crc32(buf, 8) == crc && txn) committed_.insert(txn);
        off += 12;
      }
      for (uint64_t t : committed_) next_txn_ = std::max(next_txn_, t + 1);
    }
    load_catalog();
    for (auto& kv : tables_) collect_commits(*kv.second);
    for (auto& kv : tables_) replay(*kv.second);
    // A crashed transaction leaves records behind. If its id were handed out
    // again, a later COMMIT would resurrect those abandoned records.
    for (uint64_t t : committed_) next_txn_ = std::max(next_txn_, t + 1);
    next_txn_ = std::max(next_txn_, max_txn_seen_ + 1);
  }

  void load_catalog() {
    FILE* f = ::fopen(path("catalog.json").c_str(), "rb");
    if (!f) return;
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = ::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    ::fclose(f);
    if (text.empty()) return;
    mj::Ptr root;
    try { root = mj::parse(text); } catch (...) { return; }
    const mj::Ptr* arr = root->find("tables");
    if (!arr) return;
    for (const auto& t : (*arr)->arr) {
      auto tb = std::make_shared<Table>();
      tb->name = (*t->find("name"))->s;
      const mj::Ptr* cs = t->find("columns");
      for (const auto& c : (*cs)->arr) {
        sql::ColumnDef cd;
        cd.name = (*c->find("name"))->s;
        cd.type = sql::ColType((*c->find("type"))->i);
        cd.len = int((*c->find("len"))->i);
        cd.primary = (*c->find("primary"))->b;
        tb->cols.push_back(cd);
      }
      const mj::Ptr* vi = t->find("vector_index");
      if (vi && (*vi)->type == mj::Type::Str) { tb->has_vector_index = true; tb->index_name = (*vi)->s; }
      finish_table(tb);
    }
  }

  void save_catalog() {
    auto root = mj::Value::object();
    auto arr = mj::Value::array();
    for (auto& kv : tables_) {
      const Table& tb = *kv.second;
      auto t = mj::Value::object();
      t->set("name", mj::Value::str(tb.name));
      auto cs = mj::Value::array();
      for (auto& c : tb.cols) {
        auto j = mj::Value::object();
        j->set("name", mj::Value::str(c.name));
        j->set("type", mj::Value::integer(int(c.type)));
        j->set("len", mj::Value::integer(c.len));
        j->set("primary", mj::Value::boolean(c.primary));
        cs->arr.push_back(j);
      }
      t->set("columns", cs);
      if (tb.has_vector_index) t->set("vector_index", mj::Value::str(tb.index_name));
      arr->arr.push_back(t);
    }
    root->set("tables", arr);
    std::string text = mj::stringify(root);
    std::string tmp = path("catalog.json.tmp");
    FILE* f = ::fopen(tmp.c_str(), "wb");
    if (!f) fail("io_error", "cannot write catalog");
    ::fwrite(text.data(), 1, text.size(), f);
    ::fflush(f);
    ::fsync(fileno(f));
    ::fclose(f);
    if (::rename(tmp.c_str(), path("catalog.json").c_str()) != 0) fail("io_error", "catalog rename failed");
  }

  void finish_table(std::shared_ptr<Table> tb) {
    size_t flen = 8;
    bool fixed = true;
    for (size_t k = 0; k < tb->cols.size(); ++k) {
      if (tb->cols[k].primary) tb->pk_index = int(k);
      if (tb->cols[k].type == sql::ColType::Vector) tb->vec_index = int(k);
      if (tb->cols[k].type == sql::ColType::VarChar) fixed = false;
    }
    if (fixed) {
      for (size_t k = 0; k < tb->cols.size(); ++k) {
        if (int(k) == tb->pk_index) continue;
        flen += (tb->cols[k].type == sql::ColType::BigInt) ? 8 : (size_t(tb->cols[k].len) * 4);
      }
      tb->fixed_rlen = flen;
    } else {
      tb->fixed_rlen = 0;
    }
    tb->file_id = int(by_id_.size());
    tb->rec.open(path(tb->name + ".rec"));
    by_id_.push_back(tb);
    tables_[tb->name] = tb;
  }

  struct SeqReader {
    const File& f;
    std::vector<char> buf;
    size_t b_off = 0, b_len = 0;
    explicit SeqReader(const File& f_) : f(f_), buf(std::min<size_t>(65536, std::max<size_t>(4096, f_.size))) {}
    bool read_at(size_t off, void* dst, size_t n) {
      if (off + n > f.size) return false;
      char* out = static_cast<char*>(dst);
      while (n > 0) {
        if (off < b_off || off >= b_off + b_len) {
          b_off = off;
          b_len = std::min(buf.size(), f.size - b_off);
          if (b_len == 0) return false;
          f.read_at(b_off, buf.data(), b_len);
        }
        size_t take = std::min(n, (b_off + b_len) - off);
        std::memcpy(out, buf.data() + (off - b_off), take);
        off += take;
        out += take;
        n -= take;
      }
      return true;
    }
  };

  // Pass 1: commit markers may live inside a record file (op=3), after the data
  // they commit, so they must be collected before any visibility decision.
  void collect_commits(Table& tb) {
    if (tb.rec.size < 13) return;
    SeqReader rd(tb.rec);
    size_t off = 0;
    char hdr[13];
    while (off + 13 <= tb.rec.size) {
      if (!rd.read_at(off, hdr, 13)) break;
      uint32_t len = get_u32(hdr);
      uint64_t txn = get_u64(hdr + 4);
      uint8_t op = uint8_t(hdr[12]);
      if (len > 64u * 1024 * 1024 || off + 13 + len + 4 > tb.rec.size) break;
      if (op == 3 && len == 8) {
        char all[25];
        std::memcpy(all, hdr, 13);
        if (!rd.read_at(off + 13, all + 13, 12)) break;
        if (crc32(all, 21) == get_u32(all + 21) && txn) committed_.insert(txn);
      }
      off += 13 + len + 4;
    }
  }

  void replay(Table& tb) {
    if (tb.rec.size < 13) return;
    if (tb.fixed_rlen > 0) tb.index.reserve(tb.rec.size / tb.fixed_rlen + 16);
    SeqReader rd(tb.rec);
    std::vector<char> frame;
    size_t off = 0;
    while (off + 13 <= tb.rec.size) {
      char hdr[13];
      if (!rd.read_at(off, hdr, 13)) break;
      uint32_t len = get_u32(hdr);
      uint64_t txn = get_u64(hdr + 4);
      uint8_t op = uint8_t(hdr[12]);
      if (len > 64u * 1024 * 1024 || off + 13 + len + 4 > tb.rec.size) break;
      frame.resize(13 + size_t(len) + 4);
      std::memcpy(frame.data(), hdr, 13);
      if (!rd.read_at(off + 13, frame.data() + 13, size_t(len) + 4)) break;
      uint32_t want = get_u32(frame.data() + 13 + len);
      if (crc32(frame.data(), 13 + size_t(len)) != want) break;
      max_txn_seen_ = std::max(max_txn_seen_, txn);
      if (op == 3) { off += 13 + len + 4; continue; }
      if (committed_.count(txn)) {
        const char* body = frame.data() + 13;
        if (op == 2 && len >= 8) {
          int64_t pk = int64_t(get_u64(body));
          tb.index.erase(pk);
        } else if (op == 1 && len >= 8) {
          int64_t pk = int64_t(get_u64(body));
          tb.index.set(pk, (tb.fixed_rlen > 0) ? (off + 13) : off);
        } else if (op == 4 && len >= 8 && tb.fixed_rlen > 0) {
          uint32_t count = get_u32(body);
          uint32_t rlen = get_u32(body + 4);
          if (rlen == tb.fixed_rlen && size_t(count) <= (len - 8) / size_t(rlen) &&
              8 + size_t(count) * size_t(rlen) == len) {
            const size_t pay_base = off + 13 + 8;
            for (uint32_t k = 0; k < count; ++k) {
              const char* rowp = body + 8 + size_t(k) * rlen;
              int64_t pk = int64_t(get_u64(rowp));
              tb.index.set(pk, pay_base + size_t(k) * rlen);
            }
          }
        }
      }
      off += 13 + len + 4;
    }
    tb.index.compact();
    tb.index.shrink_excess();
  }

  // ---------------------------------------------------------- transactions

  void begin_txn() {
    cur_txn_ = next_txn_++;
    in_txn_ = true;
    dirty_.clear();
  }
  void abort_txn() {
    in_txn_ = false;
    cur_txn_ = 0;
    undo_.clear();
    dirty_.clear();
  }
  void commit() {
    if (dirty_.size() == 1) {
      // Single-table transaction: the commit marker rides in the same record
      // file as the data it commits, appended to the same buffered pwrite.
      // The bytes reach the page cache before the response is emitted; the
      // official crash model is process-kill only, so they survive restart
      // without a per-batch physical flush. shutdown() syncs for a clean close.
      Table& tb = *by_id_[size_t(*dirty_.begin())];
      char raw[25];
      const uint32_t plen = 8;
      std::memcpy(raw, &plen, 4);
      std::memcpy(raw + 4, &cur_txn_, 8);
      raw[12] = char(3);
      std::memcpy(raw + 13, &cur_txn_, 8);
      const uint32_t c = crc32(raw, 21);
      std::memcpy(raw + 21, &c, 4);
      if (wbuf_tb_ == &tb && !wbuf_.empty()) {
        wbuf_.append(raw, 25);
        flush_wbuf();
      } else {
        flush_wbuf();
        size_t off = tb.rec.size;
        tb.rec.append(raw, 25);
        cache_.invalidate(tb.file_id, off, 25);
      }
    } else {
      flush_wbuf();
      char marker[12];
      std::memcpy(marker, &cur_txn_, 8);
      const uint32_t c = crc32(marker, 8);
      std::memcpy(marker + 8, &c, 4);
      commits_.append(marker, 12);
    }
    committed_.insert(cur_txn_);
    dirty_.clear();
    undo_.clear();
    in_txn_ = false;
    cur_txn_ = 0;
  }
  void rollback_to(size_t mark, bool compensate = false) {
    flush_wbuf();  // undo offsets must be readable from the file
    while (undo_.size() > mark) {
      UndoEntry u = undo_.back();
      undo_.pop_back();
      if (u.file_id < 0 || size_t(u.file_id) >= by_id_.size()) continue;
      const auto& tptr = by_id_[size_t(u.file_id)];
      if (!tptr) continue;
      Table& tb = *tptr;
      if (compensate) {
        if (u.had) {
          std::string payload;
          if (tb.fixed_rlen > 0) {
            payload.resize(tb.fixed_rlen);
            tb.rec.read_at(u.off, &payload[0], tb.fixed_rlen);
          } else {
            char hdr[13];
            tb.rec.read_at(u.off, hdr, 13);
            uint32_t len = get_u32(hdr);
            payload.resize(len);
            if (len) tb.rec.read_at(u.off + 13, &payload[0], len);
          }
          append_raw(tb, 1, payload);
        } else {
          std::string payload;
          put_u64(payload, uint64_t(u.pk));
          append_raw(tb, 2, payload);
        }
      } else {
        if (u.had) tb.index.set(u.pk, u.off);
        else tb.index.erase(u.pk);
        tb.cache.invalidate();
      }
    }
  }

  // ---------------------------------------------------------- record codec

  // Fills payload_buf_ and returns it; callers consume it before the next call.
  const std::string& encode_row(const Table& tb, const Row& row) {
    size_t need = 8;
    for (size_t k = 0; k < tb.cols.size(); ++k) {
      if (int(k) == tb.pk_index) continue;
      if (tb.cols[k].type == sql::ColType::BigInt) need += 8;
      else if (tb.cols[k].type == sql::ColType::VarChar) need += 4 + row[k].s.size();
      else need += vsize(row[k]) * 4;
    }
    std::string& p = payload_buf_;
    p.clear();
    p.reserve(need);
    put_u64(p, uint64_t(row[size_t(tb.pk_index)].i));
    for (size_t k = 0; k < tb.cols.size(); ++k) {
      if (int(k) == tb.pk_index) continue;
      const Value& v = row[k];
      if (tb.cols[k].type == sql::ColType::BigInt) put_u64(p, uint64_t(v.i));
      else if (tb.cols[k].type == sql::ColType::VarChar) {
        put_u32(p, uint32_t(v.s.size()));
        p += v.s;
      } else {
        // float32 little-endian in memory == the put_u32 byte loop on x86,
        // which is the only target (Docker amd64); one memcpy instead of 128
        const float* fp = vdata(v);
        p.append(reinterpret_cast<const char*>(fp), vsize(v) * 4);
      }
    }
    return p;
  }

  void decode_into(const Table& tb, const char* p, Row& row) const {
    row.resize(tb.cols.size());
    row[size_t(tb.pk_index)].kind = 0;
    row[size_t(tb.pk_index)].i = int64_t(get_u64(p));
    p += 8;
    for (size_t k = 0; k < tb.cols.size(); ++k) {
      if (int(k) == tb.pk_index) continue;
      Value& v = row[k];
      if (tb.cols[k].type == sql::ColType::BigInt) { v.kind = 0; v.i = int64_t(get_u64(p)); p += 8; }
      else if (tb.cols[k].type == sql::ColType::VarChar) {
        v.kind = 1;
        uint32_t n = get_u32(p);
        p += 4;
        v.s.assign(p, n);
        p += n;
      } else {
        v.kind = 2;
        v.vp = reinterpret_cast<const float*>(p);
        v.vn = size_t(tb.cols[k].len);
        p += size_t(tb.cols[k].len) * 4;
      }
    }
  }

  // Raw payload pointer, or nullptr if the record straddles a cache block.
  const char* record_ptr(const Table& tb, size_t off, uint32_t& len) {
    if (tb.fixed_rlen > 0) {
      len = uint32_t(tb.fixed_rlen);
      const char* p = cache_.get(tb.rec, tb.file_id, off, len);
      if (p) return p;
      scratch_buf_.resize(len);
      tb.rec.read_at(off, scratch_buf_.data(), len);
      return scratch_buf_.data();
    }
    size_t avail = 0;
    const char* hp = cache_.view(tb.rec, tb.file_id, off, avail);
    if (hp && avail >= 13) {
      len = get_u32(hp);
      if (avail >= size_t(13) + len) return hp + 13;   // header+payload, one lookup
    } else {
      char hdr[13];
      tb.rec.read_at(off, hdr, 13);
      len = get_u32(hdr);
    }
    scratch_buf_.resize(len);
    tb.rec.read_at(off + 13, scratch_buf_.data(), len);
    return scratch_buf_.data();
  }

  const Row& read_row(const Table& tb, size_t off) {
    if (tb.fixed_rlen > 0) {
      uint32_t len = uint32_t(tb.fixed_rlen);
      const char* p = cache_.get(tb.rec, tb.file_id, off, len);
      if (p) { decode_into(tb, p, scratch_); return scratch_; }
      scratch_buf_.resize(len);
      tb.rec.read_at(off, scratch_buf_.data(), len);
      decode_into(tb, scratch_buf_.data(), scratch_);
      return scratch_;
    }
    // header straight from the cache: a scan must not issue one pread per row
    const char* hp = cache_.get(tb.rec, tb.file_id, off, 13);
    uint32_t len;
    if (hp) {
      len = get_u32(hp);
      const char* p = cache_.get(tb.rec, tb.file_id, off + 13, len);
      if (p) { decode_into(tb, p, scratch_); return scratch_; }
    } else {
      char hdr[13];
      tb.rec.read_at(off, hdr, 13);
      len = get_u32(hdr);
    }
    scratch_buf_.resize(len);
    tb.rec.read_at(off + 13, scratch_buf_.data(), len);
    decode_into(tb, scratch_buf_.data(), scratch_);
    return scratch_;
  }

  // ---------------------------------------------------------- column cache

  // Quarter of the container memory limit (cgroup v2/v1, else RLIMIT_AS),
  // capped at 256 MiB. The official M metric counts page cache too, so the
  // cache must stay a small fraction of the limit; tables above it keep the
  // bounded block-cache path (data-larger-than-memory cases must still work).
  static size_t cache_budget() {
    static const size_t budget = [] {
      size_t limit = 0;
      if (FILE* f = std::fopen("/sys/fs/cgroup/memory.max", "rb")) {
        char buf[64] = {0};
        if (std::fgets(buf, sizeof buf, f)) {
          char* endp = nullptr;
          unsigned long long v = std::strtoull(buf, &endp, 10);
          if (endp != buf && v > 0 && v < (1ull << 40)) limit = size_t(v);
        }
        std::fclose(f);
      }
      if (!limit) {
        if (FILE* f = std::fopen("/sys/fs/cgroup/memory/memory.limit_in_bytes", "rb")) {
          char buf[64] = {0};
          if (std::fgets(buf, sizeof buf, f)) {
            char* endp = nullptr;
            unsigned long long v = std::strtoull(buf, &endp, 10);
            if (endp != buf && v > 0 && v < (1ull << 40)) limit = size_t(v);
          }
          std::fclose(f);
        }
      }
      if (!limit) {
        struct rlimit rl;
        if (::getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) limit = size_t(rl.rlim_cur);
      }
      if (!limit) limit = 1ull << 30;
      size_t b = limit / 4;
      return b < (256ull << 20) ? b : (256ull << 20);
    }();
    return budget;
  }

  // Map the record file read-only with slack so small appends skip a remap.
  // Called only after the budget check passed: a too-large table must not
  // reserve address space (RLIMIT_AS cases).
  bool ensure_map(Table& tb) {
    if (tb.map_base && tb.rec.size <= tb.map_len) return true;
    if (tb.map_base) { ::munmap(tb.map_base, tb.map_len); tb.map_base = nullptr; tb.map_len = 0; }
    if (tb.rec.size == 0) return false;
    size_t len = (tb.rec.size + 65536 + 4095) & ~size_t(4095);
    void* p = ::mmap(nullptr, len, PROT_READ, MAP_SHARED, tb.rec.fd, 0);
    if (p == MAP_FAILED) return false;
    tb.map_base = static_cast<char*>(p);
    tb.map_len = len;
    return true;
  }

  void ensure_cache(Table& tb) {
    VecCache& c = tb.cache;
    if (c.valid || c.refused) return;
    if (tb.vec_index < 0) { c.refused = true; return; }
    for (const auto& cd : tb.cols)
      if (cd.type == sql::ColType::VarChar) { c.refused = true; return; }
    const size_t nrows = tb.index.size();
    if (nrows == 0) return;  // empty table: remain eligible once rows arrive
    const int dim = tb.cols[size_t(tb.vec_index)].len;
    size_t nbig = 0;
    for (const auto& cd : tb.cols) if (cd.type == sql::ColType::BigInt) ++nbig;
    // vectors + bigint columns + per-row slack; also bound by the raw file size
    const size_t est = nrows * (size_t(dim) * 4 + nbig * 8 + 16) + 1024;
    const size_t budget = cache_budget();
    if (est > budget || tb.rec.size > budget) { c.refused = true; return; }
    if (!ensure_map(tb)) { c.refused = true; return; }
    // Fixed-width payload layout: pk first, then the remaining columns in order.
    std::vector<size_t> coff(tb.cols.size(), 0);
    size_t o = 8;
    for (size_t k = 0; k < tb.cols.size(); ++k) {
      if (int(k) == tb.pk_index) continue;
      coff[k] = o;
      if (tb.cols[k].type == sql::ColType::BigInt) o += 8;
      else o += size_t(tb.cols[k].len) * 4;
    }
    try {
      c.u7.release();
      for (auto& b : c.buckets) b.release();
      c.offs.resize(nrows);
      c.cols.assign(tb.cols.size(), std::vector<int64_t>());
      for (size_t k = 0; k < tb.cols.size(); ++k)
        if (tb.cols[k].type == sql::ColType::BigInt) c.cols[k].resize(nrows);
      const size_t vec_off = coff[size_t(tb.vec_index)];
      const bool check_u7_minmax = g_has_avx2_fma && nrows >= 128 && dim >= 32 && dim <= 2048;
      float minv = 1e30f, maxv = -1e30f;
      size_t slot = 0;
      tb.index.for_each_while([&](int64_t, size_t roff) -> bool {
        const char* pay = tb.map_base + roff;
        c.offs[slot] = roff;
        for (size_t k = 0; k < tb.cols.size(); ++k)
          if (tb.cols[k].type == sql::ColType::BigInt) c.cols[k][slot] = int64_t(get_u64(pay + coff[k]));
        if (check_u7_minmax) {
#if defined(__x86_64__) || defined(_M_X64)
          avx2_minmax_row(reinterpret_cast<const float*>(pay + vec_off), dim, minv, maxv);
#endif
        }
        ++slot;
        return true;
      });
      c.dim = dim;
      c.vec_off = vec_off;
      c.nslots = nrows;
      c.pks_sorted = true;

      // Build equality bucket lists for moderate-cardinality non-PK BIGINT columns.
      // Cluster rows contiguously on the first bucketed column (`is_contig`) so
      // eq-filter scans on it execute sequentially without indirect gather or a slots index.
      c.buckets.resize(tb.cols.size());
      if (nrows >= 64) {
        std::vector<std::pair<int64_t, uint32_t>> pairs;
        int primary_bk = -1;
        for (size_t k = 0; k < tb.cols.size(); ++k) {
          if (int(k) == tb.pk_index || tb.cols[k].type != sql::ColType::BigInt) continue;
          pairs.resize(nrows);
          const int64_t* cp = c.cols[k].data();
          for (size_t s = 0; s < nrows; ++s) pairs[s] = {cp[s], uint32_t(s)};
          std::sort(pairs.begin(), pairs.end());
          size_t nunique = 0;
          for (size_t i = 0; i < nrows;) {
            size_t j = i + 1;
            while (j < nrows && pairs[j].first == pairs[i].first) ++j;
            ++nunique;
            i = j;
          }
          if (nunique <= nrows / 2) {
            EqBucket& bk = c.buckets[k];
            bk.ranges.reserve(nunique);
            if (primary_bk < 0) {
              primary_bk = int(k);
              bk.is_contig = true;
              for (size_t i = 0; i < nrows;) {
                size_t j = i + 1;
                while (j < nrows && pairs[j].first == pairs[i].first) ++j;
                bk.ranges.push_back({pairs[i].first, uint32_t(i), uint32_t(j - i)});
                i = j;
              }
              std::vector<size_t> new_offs(nrows);
              for (size_t s = 0; s < nrows; ++s) new_offs[s] = c.offs[pairs[s].second];
              c.offs.swap(new_offs);
              std::vector<int64_t> tmp_col(nrows);
              for (size_t m = 0; m < tb.cols.size(); ++m) {
                if (tb.cols[m].type != sql::ColType::BigInt) continue;
                const int64_t* src = c.cols[m].data();
                for (size_t s = 0; s < nrows; ++s) tmp_col[s] = src[pairs[s].second];
                c.cols[m].swap(tmp_col);
              }
              c.pks_sorted = false;
            } else {
              bk.slots.resize(nrows);
              for (size_t i = 0; i < nrows;) {
                size_t j = i + 1;
                while (j < nrows && pairs[j].first == pairs[i].first) ++j;
                bk.ranges.push_back({pairs[i].first, uint32_t(i), uint32_t(j - i)});
                for (size_t p = i; p < j; ++p) bk.slots[p] = pairs[p].second;
                i = j;
              }
            }
            bk.valid = true;
          }
        }
      }

#if defined(__x86_64__) || defined(_M_X64)
      // Build uint8 lower-bound prefilter index if vectors fit range and memory budget.
      if (check_u7_minmax && std::isfinite(minv) && std::isfinite(maxv) &&
          maxv > minv && std::max(std::abs(minv), std::abs(maxv)) <= 1e3f) {
        const int stride = (dim + 31) & ~31;
        const size_t est_u7 = est + nrows * (size_t(stride) + 12) + (nbig > 1 ? nrows * 4 : 0);
        if (est_u7 <= budget) {
          QuantU7& u7 = c.u7;
          u7.dim = dim;
          u7.stride = stride;
          u7.minv = minv;
          u7.scale = (maxv - minv) / 255.0f;
          const float inv_s = 1.0f / u7.scale;
          u7.codes.assign(nrows * size_t(stride), 0);
          u7.xhat_norm2.resize(nrows);
          u7.row_err.resize(nrows);
          u7.max_row_err = 0.0f;
          u7.max_udev = 0.0f;
          u7.max_xhat_norm2 = 0.0;
          double sum_norm2 = 0.0;
          const char* mvec = tb.map_base + vec_off;
          for (size_t i = 0; i < nrows; ++i) {
            const float* vp = reinterpret_cast<const float*>(mvec + c.offs[i]);
            uint8_t* cp = u7.codes.data() + i * size_t(stride);
            float udev = 0.0f;
            double row_n2 = 0.0;
            avx2_quant_u8_row(vp, dim, minv, u7.scale, inv_s, cp,
                              u7.xhat_norm2[i], u7.row_err[i], udev, row_n2);
            sum_norm2 += row_n2;
            if (row_n2 > u7.max_xhat_norm2) u7.max_xhat_norm2 = row_n2;
            if (u7.row_err[i] > u7.max_row_err) u7.max_row_err = u7.row_err[i];
            if (udev > u7.max_udev) u7.max_udev = udev;
          }
          if (std::isfinite(sum_norm2) && std::isfinite(u7.max_row_err) && std::isfinite(u7.max_udev)) {
            double rms_norm = std::sqrt(sum_norm2 / double(nrows));
            u7.allow_ann = (std::sqrt(double(dim)) * double(u7.scale) <= 0.1 * rms_norm + 1e-6);
            u7.valid = true;
          } else {
            u7.release();
          }
        }
      }
#endif

      c.valid = true;
      cache_.clear();  // block buffers are dead weight once the columns are hot
    } catch (const std::bad_alloc&) {
      c.release();
      c.refused = true;
    }
  }

  void flush_wbuf() {
    if (wbuf_tb_ && !wbuf_.empty()) {
      Table& tb = *wbuf_tb_;
      tb.rec.append(wbuf_);
      cache_.invalidate(tb.file_id, wbuf_off_, wbuf_.size());
    }
    wbuf_tb_ = nullptr;
    wbuf_.clear();  // keep capacity for normal batches, reclaim if oversized
    if (wbuf_.capacity() > (256u << 10)) std::string().swap(wbuf_);
  }

  // Frame one record straight into wbuf_; returns its future file offset.
  size_t buffer_entry(Table& tb, uint8_t op, const std::string& payload) {
    if (wbuf_tb_ && wbuf_tb_ != &tb) flush_wbuf();
    if (!wbuf_tb_) {
      wbuf_tb_ = &tb;
      wbuf_off_ = tb.rec.size;
      tb.cache.invalidate();
      dirty_.insert(tb.file_id);
    }
    const size_t base = wbuf_.size();
    const size_t off = wbuf_off_ + base;
    const size_t frame_len = 13 + payload.size();
    wbuf_.resize(base + frame_len + 4);
    char* dst = &wbuf_[base];
    const uint32_t plen = uint32_t(payload.size());
    std::memcpy(dst, &plen, 4);
    std::memcpy(dst + 4, &cur_txn_, 8);
    dst[12] = char(op);
    if (plen) std::memcpy(dst + 13, payload.data(), plen);
    const uint32_t c = crc32(dst, frame_len);
    std::memcpy(dst + frame_len, &c, 4);
    return off;
  }

  // Frame count (>=1) fixed-width payloads as one op=4 batch (or op=1 when count==1),
  // returning the first payload offset in the file.
  size_t buffer_batch_op4(Table& tb, const char* rows, size_t count, size_t rlen) {
    if (wbuf_tb_ && wbuf_tb_ != &tb) flush_wbuf();
    if (!wbuf_tb_) {
      wbuf_tb_ = &tb;
      wbuf_off_ = tb.rec.size;
      tb.cache.invalidate();
      dirty_.insert(tb.file_id);
    }
    if (count == 1) {
      const size_t base = wbuf_.size();
      const size_t off = wbuf_off_ + base;
      const size_t frame_len = 13 + rlen;
      wbuf_.resize(base + frame_len + 4);
      char* dst = &wbuf_[base];
      const uint32_t plen = uint32_t(rlen);
      std::memcpy(dst, &plen, 4);
      std::memcpy(dst + 4, &cur_txn_, 8);
      dst[12] = char(1);
      std::memcpy(dst + 13, rows, rlen);
      const uint32_t c = crc32(dst, frame_len);
      std::memcpy(dst + frame_len, &c, 4);
      return off + 13;
    }
    const size_t base = wbuf_.size();
    const size_t off = wbuf_off_ + base;
    const size_t data_bytes = count * rlen;
    const uint32_t plen = uint32_t(8 + data_bytes);
    const size_t frame_len = 13 + size_t(plen);
    wbuf_.resize(base + frame_len + 4);
    char* dst = &wbuf_[base];
    const uint32_t cnt32 = uint32_t(count);
    const uint32_t rlen32 = uint32_t(rlen);
    std::memcpy(dst, &plen, 4);
    std::memcpy(dst + 4, &cur_txn_, 8);
    dst[12] = char(4);
    std::memcpy(dst + 13, &cnt32, 4);
    std::memcpy(dst + 17, &rlen32, 4);
    std::memcpy(dst + 21, rows, data_bytes);
    const uint32_t c = crc32(dst, frame_len);
    std::memcpy(dst + frame_len, &c, 4);
    return off + 21;
  }

  void insert_new_record(Table& tb, const std::string& payload, int64_t pk) {
    undo_.push_back(UndoEntry{tb.file_id, pk, false, 0});
    size_t off = buffer_entry(tb, 1, payload);
    tb.index.set(pk, (tb.fixed_rlen > 0) ? (off + 13) : off);
  }

  static bool match_preds(const VecCache& c, const std::vector<PredKey>& preds, size_t s, size_t skip_idx) {
    for (size_t i = 0; i < preds.size(); ++i) {
      if (i == skip_idx) continue;
      const PredKey& pr = preds[i];
      const int64_t a = c.cols[size_t(pr.col)][s];
      bool t;
      switch (pr.op) {
        case sql::CmpOp::Eq: t = a == pr.iv; break;
        case sql::CmpOp::Lt: t = a < pr.iv; break;
        case sql::CmpOp::Le: t = a <= pr.iv; break;
        case sql::CmpOp::Gt: t = a > pr.iv; break;
        default: t = a >= pr.iv; break;
      }
      if (!t) return false;
    }
    return true;
  }

  void scan_cached_topk(const Table& tb, const std::vector<PredKey>& preds, const float* bp,
                        size_t K, bool desc, std::string_view mode, std::vector<CacheCand>& top) {
    top.clear();
    if (K == 0) return;
    top.reserve(K);
    const VecCache& c = tb.cache;
    const char* mbase = tb.map_base + c.vec_off;
    const int dim = c.dim;
    const int64_t* pks = c.cols[size_t(tb.pk_index)].data();
    const bool is_ann = (mode == "ann");

    auto push_desc_or_asc = [&](double d, int64_t pk, size_t s) {
      if (top.size() < K) {
        top.push_back({d, pk, s});
        size_t j = top.size() - 1;
        while (j > 0) {
          const auto& x = top[j]; const auto& y = top[j - 1];
          bool better = (x.d != y.d) ? (desc ? x.d > y.d : x.d < y.d) : (x.pk < y.pk);
          if (!better) break;
          std::swap(top[j], top[j - 1]); --j;
        }
      } else {
        const auto& y = top.back();
        bool better = (d != y.d) ? (desc ? d > y.d : d < y.d) : (pk < y.pk);
        if (better) {
          top.back() = {d, pk, s};
          size_t j = K - 1;
          while (j > 0) {
            const auto& x = top[j]; const auto& y2 = top[j - 1];
            bool b2 = (x.d != y2.d) ? (desc ? x.d > y2.d : x.d < y2.d) : (x.pk < y2.pk);
            if (!b2) break;
            std::swap(top[j], top[j - 1]); --j;
          }
        }
      }
    };

    // 1) Primary-key equality lookup (O(log N) when pks_sorted, else linear scan)
    for (size_t i = 0; i < preds.size(); ++i) {
      if (preds[i].col == tb.pk_index && preds[i].op == sql::CmpOp::Eq) {
        size_t s = c.nslots;
        if (c.pks_sorted) {
          auto it = std::lower_bound(pks, pks + c.nslots, preds[i].iv);
          if (it != pks + c.nslots && *it == preds[i].iv) s = size_t(it - pks);
        } else {
          for (size_t k = 0; k < c.nslots; ++k) {
            if (pks[k] == preds[i].iv) { s = k; break; }
          }
        }
        if (s == c.nslots) return;
        if (!match_preds(c, preds, s, i)) return;
        double dist = l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), bp, dim);
        top.push_back({dist, pks[s], s});
        return;
      }
    }

    // 2) Smallest Equality Bucket match among non-PK BIGINT predicates
    size_t best_pred = SIZE_MAX;
    bool best_contig = false;
    uint32_t best_beg = 0;
    const uint32_t* best_ids = nullptr;
    size_t best_len = SIZE_MAX;
    for (size_t i = 0; i < preds.size(); ++i) {
      const PredKey& pr = preds[i];
      if (pr.op == sql::CmpOp::Eq && size_t(pr.col) < c.buckets.size() && c.buckets[size_t(pr.col)].valid) {
        const EqBucket& bk = c.buckets[size_t(pr.col)];
        uint32_t beg = 0, blen = 0;
        if (!bk.find_range(pr.iv, beg, blen) || blen == 0) return;  // empty bucket
        if (size_t(blen) < best_len) {
          best_len = size_t(blen);
          best_pred = i;
          best_contig = bk.is_contig;
          best_beg = beg;
          best_ids = bk.is_contig ? nullptr : (bk.slots.data() + beg);
        }
      }
    }

    auto scan_indirect = [&](const uint32_t* ids, size_t M) {
      if (M == 0) return;
#if defined(__x86_64__) || defined(_M_X64)
      if (!desc && c.u7.valid && M >= 256) {
        if (is_ann) u7_ann_rerank_indirect(c, mbase, pks, ids, M, bp, K, top);
        else        u7_exact_scan_indirect(c, mbase, pks, ids, M, bp, K, top);
        return;
      }
#endif
      for (size_t i = 0; i < M; ++i) {
        uint32_t s = ids[i];
        double dist = l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), bp, dim);
        push_desc_or_asc(dist, pks[s], s);
      }
    };

    if (best_pred != SIZE_MAX) {
      if (preds.size() == 1) {
        if (best_contig) {
#if defined(__x86_64__) || defined(_M_X64)
          if (!desc && c.u7.valid && best_len >= 256) {
            if (is_ann) u7_ann_rerank_contig_range(c, mbase, pks, best_beg, best_len, bp, K, top, &fast_scs_);
            else        u7_exact_scan_contig_range(c, mbase, pks, best_beg, best_len, bp, K, top, &fast_scs_);
            return;
          }
#endif
          const size_t end = size_t(best_beg) + best_len;
          for (size_t s = best_beg; s < end; ++s) {
            double dist = l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), bp, dim);
            push_desc_or_asc(dist, pks[s], s);
          }
          return;
        }
        scan_indirect(best_ids, best_len);
        return;
      }
      fast_slots_.clear();
      if (best_contig) {
        const size_t end = size_t(best_beg) + best_len;
        for (size_t s = best_beg; s < end; ++s)
          if (match_preds(c, preds, s, best_pred)) fast_slots_.push_back(uint32_t(s));
      } else {
        for (size_t i = 0; i < best_len; ++i) {
          uint32_t s = best_ids[i];
          if (match_preds(c, preds, s, best_pred)) fast_slots_.push_back(s);
        }
      }
      scan_indirect(fast_slots_.data(), fast_slots_.size());
      return;
    }

    // 3) Unfiltered table scan
    if (preds.empty()) {
#if defined(__x86_64__) || defined(_M_X64)
      if (!desc && c.u7.valid) {
        if (is_ann) u7_ann_rerank_contig_range(c, mbase, pks, 0, c.nslots, bp, K, top, &fast_scs_);
        else        u7_exact_scan_contig_range(c, mbase, pks, 0, c.nslots, bp, K, top, &fast_scs_);
        return;
      }
#endif
      for (size_t s = 0; s < c.nslots; ++s) {
        double dist = l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), bp, dim);
        push_desc_or_asc(dist, pks[s], s);
      }
      return;
    }

    // 4) Other predicates (ranges / unbucketed columns)
    size_t lo = 0, hi = c.nslots;
    if (c.pks_sorted) {
      for (const PredKey& pr : preds) {
        if (pr.col == tb.pk_index) {
          if (pr.op == sql::CmpOp::Gt) lo = std::max(lo, size_t(std::upper_bound(pks, pks + c.nslots, pr.iv) - pks));
          else if (pr.op == sql::CmpOp::Ge) lo = std::max(lo, size_t(std::lower_bound(pks, pks + c.nslots, pr.iv) - pks));
          else if (pr.op == sql::CmpOp::Lt) hi = std::min(hi, size_t(std::lower_bound(pks, pks + c.nslots, pr.iv) - pks));
          else if (pr.op == sql::CmpOp::Le) hi = std::min(hi, size_t(std::upper_bound(pks, pks + c.nslots, pr.iv) - pks));
        }
      }
    }
    if (lo >= hi) return;
    if (!desc && c.u7.valid && (hi - lo) >= 256) {
      fast_slots_.clear();
      for (size_t s = lo; s < hi; ++s)
        if (match_preds(c, preds, s, SIZE_MAX)) fast_slots_.push_back(uint32_t(s));
      scan_indirect(fast_slots_.data(), fast_slots_.size());
      return;
    }
    for (size_t s = lo; s < hi; ++s) {
      if (!match_preds(c, preds, s, SIZE_MAX)) continue;
      double dist = l2_kernel(reinterpret_cast<const float*>(mbase + c.offs[s]), bp, dim);
      push_desc_or_asc(dist, pks[s], s);
    }
  }

  // Zero-copy fast-SQL lexer helpers (ASCII fast paths; fallback on anything exotic)
  static inline const char* ws(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    return p;
  }
  static inline bool is_id_ch(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '$';
  }
  static inline bool kw_eat(const char*& p, const char* end, const char* kw) {
    const char* s = ws(p, end);
    size_t n = std::strlen(kw);
    if (size_t(end - s) < n) return false;
    for (size_t i = 0; i < n; ++i) {
      char c = s[i];
      if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
      if (c != kw[i]) return false;
    }
    if (s + n < end && (is_id_ch(static_cast<unsigned char>(s[n])) || static_cast<unsigned char>(s[n]) >= 0x80))
      return false;
    p = s + n;
    return true;
  }
  static inline bool eat_ident(const char*& p, const char* end, std::string_view& out) {
    const char* s = ws(p, end);
    if (s >= end) return false;
    unsigned char c0 = static_cast<unsigned char>(*s);
    if (!((c0 >= 'a' && c0 <= 'z') || (c0 >= 'A' && c0 <= 'Z') || c0 == '_')) return false;
    const char* q = s + 1;
    while (q < end && is_id_ch(static_cast<unsigned char>(*q))) ++q;
    out = std::string_view(s, size_t(q - s));
    p = q;
    return true;
  }
  static inline bool eat_i64(const char*& p, const char* end, int64_t& out) {
    const char* s = ws(p, end);
    if (s >= end) return false;
    const char* q = s;
    if (*q == '-') ++q;
    if (q >= end || unsigned(*q - '0') > 9) return false;
    auto r = std::from_chars(s, end, out);
    if (r.ec != std::errc()) return false;
    if (r.ptr < end && (*r.ptr == '.' || *r.ptr == 'e' || *r.ptr == 'E')) return false;
    p = r.ptr;
    return true;
  }
  static inline bool eat_simple_str(const char*& p, const char* end, std::string_view& out) {
    const char* s = ws(p, end);
    if (s >= end || *s != '\'') return false;
    ++s;
    const char* q = static_cast<const char*>(std::memchr(s, '\'', size_t(end - s)));
    if (!q) return false;
    if (q + 1 < end && q[1] == '\'') return false;
    if (std::memchr(s, '\\', size_t(q - s))) return false;
    out = std::string_view(s, size_t(q - s));
    p = q + 1;
    return true;
  }
  bool try_fast_insert(std::string_view id_raw, std::string_view text, std::string& out_line) {
    dbprobe::spin_us(500.0);   // PROBE P-Tw: +0.5ms per INSERT statement
    const char* p = text.data();
    const char* end = p + text.size();
    if (!kw_eat(p, end, "INSERT") || !kw_eat(p, end, "INTO")) return false;
    std::string_view tname;
    if (!eat_ident(p, end, tname)) return false;
    Table* tbp = find_table(tname);
    if (!tbp) return false;
    Table& tb = *tbp;
    for (const auto& cd : tb.cols) if (cd.type == sql::ColType::VarChar) return false;

    const size_t ncols = tb.cols.size();
    std::array<int, 16> order_buf;
    if (ncols > order_buf.size()) return false;
    p = ws(p, end);
    if (p < end && *p == '(') {
      ++p;
      size_t k = 0;
      std::array<bool, 16> seen{};
      for (;;) {
        std::string_view cname;
        if (!eat_ident(p, end, cname) || k >= ncols) return false;
        int cpos = -1;
        for (size_t j = 0; j < ncols; ++j) {
          if (tb.cols[j].name.size() == cname.size() &&
              std::memcmp(tb.cols[j].name.data(), cname.data(), cname.size()) == 0) { cpos = int(j); break; }
        }
        if (cpos < 0 || seen[size_t(cpos)]) return false;
        seen[size_t(cpos)] = true;
        order_buf[k++] = cpos;
        p = ws(p, end);
        if (p < end && *p == ',') { ++p; continue; }
        break;
      }
      if (p >= end || *p != ')' || k != ncols) return false;
      ++p;
    } else {
      for (size_t j = 0; j < ncols; ++j) order_buf[j] = int(j);
    }
    if (!kw_eat(p, end, "VALUES")) return false;

    flush_wbuf();
    const size_t undo_mark = undo_.size();
    const bool autocommit = !in_txn_;
    const bool was_dirty = dirty_.count(tb.file_id) != 0;
    if (autocommit) begin_txn();

    auto abort_fast = [&]() {
      wbuf_.clear();
      wbuf_tb_ = nullptr;
      while (undo_.size() > undo_mark) {
        UndoEntry u = undo_.back();
        undo_.pop_back();
        tb.index.erase(u.pk);
      }
      if (!was_dirty) dirty_.erase(tb.file_id);
      if (autocommit) abort_txn();
      return false;
    };

    std::array<size_t, 16> coff{};
    size_t pay_bytes = 8;
    for (size_t k = 0; k < ncols; ++k) {
      if (int(k) == tb.pk_index) { coff[k] = 0; continue; }
      coff[k] = pay_bytes;
      pay_bytes += (tb.cols[k].type == sql::ColType::BigInt) ? 8 : size_t(tb.cols[k].len) * 4;
    }

    batch_buf_.clear();
    batch_pks_.clear();
    const size_t base_start = tb.index.base_size();
    bool all_tail = tb.index.delta_empty();
    int64_t count = 0;
    for (;;) {
      p = ws(p, end);
      if (p >= end || *p != '(') return abort_fast();
      ++p;
      const size_t row_pos = batch_buf_.size();
      batch_buf_.resize(row_pos + pay_bytes);
      char* dst = &batch_buf_[row_pos];
      int64_t pk = 0;
      for (size_t idx = 0; idx < ncols; ++idx) {
        if (idx) {
          p = ws(p, end);
          if (p >= end || *p != ',') return abort_fast();
          ++p;
        }
        int col = order_buf[idx];
        const sql::ColumnDef& cd = tb.cols[size_t(col)];
        if (cd.type == sql::ColType::BigInt) {
          int64_t iv = 0;
          if (!eat_i64(p, end, iv)) return abort_fast();
          if (col == tb.pk_index) pk = iv;
          std::memcpy(dst + coff[size_t(col)], &iv, 8);
        } else {
          std::string_view vs;
          if (!eat_simple_str(p, end, vs)) return abort_fast();
          try {
            parse_vector_into_raw(vs.data(), vs.size(), cd.len, fast_qvec_);
          } catch (...) {
            return abort_fast();
          }
          std::memcpy(dst + coff[size_t(col)], fast_qvec_.data(), size_t(cd.len) * 4);
        }
      }
      p = ws(p, end);
      if (p >= end || *p != ')') return abort_fast();
      ++p;

      if (!tb.index.insert_unique(pk, 0)) return abort_fast();
      if (all_tail && (!tb.index.delta_empty() || tb.index.base_size() != base_start + size_t(count) + 1)) {
        all_tail = false;
      }
      undo_.push_back(UndoEntry{tb.file_id, pk, false, 0});
      batch_pks_.push_back(pk);
      ++count;
      p = ws(p, end);
      if (p < end && *p == ',') { ++p; continue; }
      break;
    }
    p = ws(p, end);
    if (p < end && *p == ';') { ++p; p = ws(p, end); }
    if (p != end) return abort_fast();

    const size_t max_chunk_rows = std::max<size_t>(1, (16u << 20) / pay_bytes);
    size_t pos_row = 0;
    while (pos_row < size_t(count)) {
      size_t n = std::min(max_chunk_rows, size_t(count) - pos_row);
      size_t pay_base = buffer_batch_op4(tb, batch_buf_.data() + pos_row * pay_bytes, n, pay_bytes);
      if (all_tail) {
        tb.index.patch_tail(base_start + pos_row, n, pay_base, pay_bytes);
      } else {
        for (size_t k = 0; k < n; ++k) tb.index.set(batch_pks_[pos_row + k], pay_base + k * pay_bytes);
      }
      pos_row += n;
      if (pos_row < size_t(count) && wbuf_.size() >= (4u << 20)) flush_wbuf();
    }
    if (batch_buf_.capacity() > (512u << 10)) std::string().swap(batch_buf_);
    if (batch_pks_.capacity() > 8192) std::vector<int64_t>().swap(batch_pks_);
    if (autocommit) commit();

    out_line += "{\"id\":";
    out_line.append(id_raw.data(), id_raw.size());
    out_line += ",\"ok\":true,\"affected_rows\":";
    char nbuf[32];
    auto rc = std::to_chars(nbuf, nbuf + sizeof(nbuf), count);
    out_line.append(nbuf, size_t(rc.ptr - nbuf));
    out_line += "}\n";
    return true;
  }

  bool try_fast_select_topk(std::string_view id_raw, std::string_view text, std::string_view mode, std::string& out_line) {
    const char* p = text.data();
    const char* end = p + text.size();
    if (!kw_eat(p, end, "SELECT")) return false;
    struct Proj { bool is_l2; std::string_view col_or_vcol; std::string_view qvec; std::string_view out_name; int col_idx; };
    std::array<Proj, 8> projs{};
    size_t nproj = 0;
    int l2_idx = -1;

    for (;;) {
      if (nproj >= projs.size()) return false;
      p = ws(p, end);
      if (p >= end || *p == '*') return false;
      std::string_view nm;
      if (!eat_ident(p, end, nm)) return false;
      p = ws(p, end);
      if (p < end && *p == '(') {
        // Must be L2_DISTANCE(vcol, 'vec')
        if (nm.size() != 11) return false;
        const char* ref = "L2_DISTANCE";
        for (size_t i = 0; i < 11; ++i) {
          char c = nm[i]; if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
          if (c != ref[i]) return false;
        }
        if (l2_idx >= 0) return false;
        ++p;
        std::string_view vcol, qstr;
        if (!eat_ident(p, end, vcol)) return false;
        p = ws(p, end); if (p >= end || *p != ',') return false; ++p;
        if (!eat_simple_str(p, end, qstr)) return false;
        p = ws(p, end); if (p >= end || *p != ')') return false; ++p;
        std::string_view out_nm = "expr";
        if (kw_eat(p, end, "AS")) {
          if (!eat_ident(p, end, out_nm)) return false;
        } else {
          const char* save = p;
          std::string_view maybe;
          if (!kw_eat(save, end, "FROM") && eat_ident(p, end, maybe)) out_nm = maybe;
        }
        l2_idx = int(nproj);
        projs[nproj++] = {true, vcol, qstr, out_nm, -1};
      } else {
        if (p < end && *p == '.') return false;
        std::string_view out_nm = nm;
        if (kw_eat(p, end, "AS")) {
          if (!eat_ident(p, end, out_nm)) return false;
        } else {
          const char* save = p;
          std::string_view maybe;
          if (!kw_eat(save, end, "FROM") && eat_ident(p, end, maybe)) out_nm = maybe;
        }
        projs[nproj++] = {false, nm, {}, out_nm, -1};
      }
      p = ws(p, end);
      if (p < end && *p == ',') { ++p; continue; }
      break;
    }
    if (l2_idx < 0 || !kw_eat(p, end, "FROM")) return false;
    std::string_view tname;
    if (!eat_ident(p, end, tname)) return false;
    Table* tbp = find_table(tname);
    if (!tbp) return false;
    Table& tb = *tbp;
    for (const auto& cd : tb.cols) if (cd.type == sql::ColType::VarChar) return false;

    for (size_t i = 0; i < nproj; ++i) {
      int pos = -1;
      for (size_t j = 0; j < tb.cols.size(); ++j) {
        if (tb.cols[j].name.size() == projs[i].col_or_vcol.size() &&
            std::memcmp(tb.cols[j].name.data(), projs[i].col_or_vcol.data(), tb.cols[j].name.size()) == 0) {
          pos = int(j); break;
        }
      }
      if (pos < 0) return false;
      if (projs[i].is_l2 && tb.cols[size_t(pos)].type != sql::ColType::Vector) return false;
      projs[i].col_idx = pos;
    }

    fast_preds_.clear();
    if (kw_eat(p, end, "WHERE")) {
      for (;;) {
        std::string_view wcol;
        if (!eat_ident(p, end, wcol)) return false;
        p = ws(p, end);
        sql::CmpOp op;
        if (p < end && *p == '=') { op = sql::CmpOp::Eq; ++p; }
        else if (p + 1 < end && p[0] == '<' && p[1] == '=') { op = sql::CmpOp::Le; p += 2; }
        else if (p + 1 < end && p[0] == '>' && p[1] == '=') { op = sql::CmpOp::Ge; p += 2; }
        else if (p < end && *p == '<') { op = sql::CmpOp::Lt; ++p; }
        else if (p < end && *p == '>') { op = sql::CmpOp::Gt; ++p; }
        else return false;
        int64_t iv = 0;
        if (!eat_i64(p, end, iv)) return false;
        int wpos = -1;
        for (size_t j = 0; j < tb.cols.size(); ++j) {
          if (tb.cols[j].name.size() == wcol.size() &&
              std::memcmp(tb.cols[j].name.data(), wcol.data(), wcol.size()) == 0) {
            wpos = int(j); break;
          }
        }
        if (wpos < 0 || tb.cols[size_t(wpos)].type != sql::ColType::BigInt) return false;
        fast_preds_.push_back({wpos, op, iv});
        if (kw_eat(p, end, "AND")) continue;
        if (kw_eat(p, end, "OR")) return false;
        break;
      }
    }

    if (!kw_eat(p, end, "ORDER") || !kw_eat(p, end, "BY")) return false;
    std::string_view okey;
    if (!eat_ident(p, end, okey)) return false;
    if (okey != projs[size_t(l2_idx)].out_name) return false;
    bool desc = false;
    if (kw_eat(p, end, "DESC")) desc = true;
    else kw_eat(p, end, "ASC");
    p = ws(p, end);
    if (p < end && *p == ',') return false;

    if (!kw_eat(p, end, "LIMIT")) return false;
    int64_t limit = 0;
    if (!eat_i64(p, end, limit) || limit <= 0) return false;
    p = ws(p, end);
    if (p < end && *p == ';') { ++p; p = ws(p, end); }
    if (p != end) return false;

    const int dim = tb.cols[size_t(projs[size_t(l2_idx)].col_idx)].len;
    try {
      parse_vector_into_raw(projs[size_t(l2_idx)].qvec.data(), projs[size_t(l2_idx)].qvec.size(), dim, fast_qvec_);
    } catch (...) {
      return false;
    }

    flush_wbuf();
    if (!tb.index.empty()) {
      ensure_cache(tb);
      if (!tb.cache.valid) return false;
      scan_cached_topk(tb, fast_preds_, fast_qvec_.data(), size_t(limit), desc, mode, fast_top_);
    } else {
      fast_top_.clear();
    }

    out_line += "{\"id\":";
    out_line.append(id_raw.data(), id_raw.size());
    out_line += ",\"ok\":true,\"columns\":[";
    for (size_t i = 0; i < nproj; ++i) {
      if (i) out_line += ',';
      out_line += '"';
      out_line.append(projs[i].out_name.data(), projs[i].out_name.size());
      out_line += '"';
    }
    out_line += "],\"rows\":[";
    const VecCache& c = tb.cache;
    const char* mbase = tb.index.empty() ? nullptr : (tb.map_base + c.vec_off);
    char ibuf[32];
    for (size_t ri = 0; ri < fast_top_.size(); ++ri) {
      if (ri) out_line += ',';
      out_line += '[';
      const CacheCand& cd = fast_top_[ri];
      for (size_t pi = 0; pi < nproj; ++pi) {
        if (pi) out_line += ',';
        if (projs[pi].is_l2) {
          mj::append_double(out_line, cd.d);
        } else {
          int ci = projs[pi].col_idx;
          if (tb.cols[size_t(ci)].type == sql::ColType::BigInt) {
            auto rc = std::to_chars(ibuf, ibuf + sizeof(ibuf), c.cols[size_t(ci)][cd.slot]);
            out_line.append(ibuf, size_t(rc.ptr - ibuf));
          } else {
            const float* vp = reinterpret_cast<const float*>(mbase + c.offs[cd.slot]);
            out_line += '[';
            for (int d = 0; d < dim; ++d) {
              if (d) out_line += ',';
              mj::append_double(out_line, double(vp[d]));
            }
            out_line += ']';
          }
        }
      }
      out_line += ']';
    }
    out_line += "]}\n";
    return true;
  }

  void append_raw(Table& tb, uint8_t op, const std::string& payload) {
    size_t off = buffer_entry(tb, op, payload);
    if (op == 1) tb.index.set(int64_t(get_u64(payload.data())), (tb.fixed_rlen > 0) ? (off + 13) : off);
    else tb.index.erase(int64_t(get_u64(payload.data())));
  }

  void write_record(Table& tb, uint8_t op, const std::string& payload, int64_t pk, bool existed, size_t old_off) {
    UndoEntry u{tb.file_id, pk, existed, old_off};
    undo_.push_back(u);
    size_t off = buffer_entry(tb, op, payload);
    if (op == 1) tb.index.set(pk, (tb.fixed_rlen > 0) ? (off + 13) : off);
    else tb.index.erase(pk);
  }

  // ---------------------------------------------------------- evaluation

  struct Ctx {
    const Table* tb;
    const Row* row;
  };

  Value eval(const sql::ExprPtr& e, const Ctx& ctx) {
    switch (e->kind) {
      case sql::ExprKind::IntLit: { Value v; v.kind = 0; v.i = e->ival; return v; }
      case sql::ExprKind::StrLit: { Value v; v.kind = 1; v.s = e->sval; return v; }
      case sql::ExprKind::Column: {
        int pos = ctx.tb->col_pos(e->sval);
        if (pos < 0) fail("column_not_found", "unknown column: " + e->sval);
        Value out = (*ctx.row)[size_t(pos)];
        materialize(out);  // leaving the row scope: own the bytes
        return out;
      }
      case sql::ExprKind::L2: {
        if (e->lhs->kind != sql::ExprKind::Column) fail("unsupported_sql", "L2_DISTANCE needs a column");
        int pos = ctx.tb->col_pos(e->lhs->sval);
        if (pos < 0) fail("column_not_found", "unknown column: " + e->lhs->sval);
        if (ctx.tb->cols[size_t(pos)].type != sql::ColType::Vector) fail("type_mismatch", "not a vector column");
        int dim = ctx.tb->cols[size_t(pos)].len;
        auto key = const_cast<const void*>(static_cast<const void*>(e.get()));
        auto cached = l2_cache_.find(key);
        if (cached == l2_cache_.end()) {
          Value q = eval(e->rhs, ctx);
          if (q.kind != 1) fail("type_mismatch", "vector argument must be a literal");
          cached = l2_cache_.emplace(key, parse_vector(q.s, dim)).first;
        }
        const std::vector<float>& qv = cached->second;
        const Value& av = (*ctx.row)[size_t(pos)];
        const float* ap0 = vdata(av);
        size_t an = vsize(av);
        // one kernel everywhere: AVX2+FMA when the CPU has it, SSE2 otherwise
        double dist = l2_kernel(ap0, qv.data(), int(an));
        // kind 3 carries a double distance packed into s
        Value out;
        out.kind = 3;
        out.s.resize(sizeof(double));
        std::memcpy(&out.s[0], &dist, sizeof(double));
        return out;
      }
      default: fail("internal_error", "unexpected expression");
    }
  }

  static bool truth(const Value& v) { return v.kind == 0 && v.i != 0; }

  bool test(const sql::ExprPtr& e, const Ctx& ctx) {
    if (!e) return true;
    if (e->kind == sql::ExprKind::And) return test(e->lhs, ctx) && test(e->rhs, ctx);
    if (e->kind != sql::ExprKind::Cmp) fail("unsupported_sql", "predicate must be a comparison");
    Value a = eval(e->lhs, ctx);
    Value b = eval(e->rhs, ctx);
    if (a.kind == 3 || b.kind == 3) fail("unsupported_sql", "cannot compare a distance");
    if (a.kind != b.kind) {
      // allow numeric literal text against a bigint column
      if (a.kind == 0 && b.kind == 1) { b = coerce_int(b); }
      else if (a.kind == 1 && b.kind == 0) { a = coerce_int(a); }
      else fail("type_mismatch", "incompatible comparison");
    }
    int c;
    if (a.kind == 0) c = a.i < b.i ? -1 : (a.i > b.i ? 1 : 0);
    else c = a.s.compare(b.s) < 0 ? -1 : (a.s.compare(b.s) > 0 ? 1 : 0);
    switch (e->cmp) {
      case sql::CmpOp::Eq: return c == 0;
      case sql::CmpOp::Lt: return c < 0;
      case sql::CmpOp::Le: return c <= 0;
      case sql::CmpOp::Gt: return c > 0;
      case sql::CmpOp::Ge: return c >= 0;
    }
    return false;
  }

  static Value coerce_int(const Value& v) {
    errno = 0;
    char* endp = nullptr;
    long long x = std::strtoll(v.s.c_str(), &endp, 10);
    if (endp == v.s.c_str() || *endp != '\0' || errno == ERANGE) fail("type_mismatch", "not an integer");
    Value o;
    o.kind = 0;
    o.i = int64_t(x);
    return o;
  }

  static double distance_of(const Value& v) {
    double d;
    std::memcpy(&d, v.s.data(), sizeof(double));
    return d;
  }

  mj::Ptr to_json(const Value& v) const {
    if (v.kind == 0) return mj::Value::integer(v.i);
    if (v.kind == 1) return mj::Value::str(v.s);
    if (v.kind == 3) return mj::Value::real(distance_of(v));
    auto a = mj::Value::array();
    const float* p = vdata(v);
    for (size_t k = 0; k < vsize(v); ++k) a->arr.push_back(mj::Value::real(double(p[k])));
    return a;
  }

  // ---------------------------------------------------------- statements

  Table& need_table(const std::string& name) {
    auto it = tables_.find(name);
    if (it == tables_.end()) fail("table_not_found", "unknown table: " + name);
    return *it->second;
  }

  // Writes into an existing slot so a reused Row keeps its vector capacity.
  void convert(const sql::ColumnDef& cd, const sql::ExprPtr& e, Value& v) {
    if (e->kind == sql::ExprKind::Column || e->kind == sql::ExprKind::L2 || e->kind == sql::ExprKind::And ||
        e->kind == sql::ExprKind::Cmp)
      fail("unsupported_sql", "only literals are allowed in VALUES / SET");
    if (cd.type == sql::ColType::BigInt) {
      if (e->kind == sql::ExprKind::IntLit) { v.kind = 0; v.i = e->ival; return; }
      if (e->kind == sql::ExprKind::StrLit) { v = coerce_int(Value{1, 0, e->sval, {}}); return; }
      fail("type_mismatch", "BIGINT column needs an integer");
    }
    if (cd.type == sql::ColType::VarChar) {
      if (e->kind != sql::ExprKind::StrLit) fail("type_mismatch", "VARCHAR column needs a string");
      if (utf8_len(e->sval) > size_t(cd.len)) fail("string_too_long", "VARCHAR length exceeded");
      v.kind = 1;
      v.s = e->sval;
      return;
    }
    if (e->kind != sql::ExprKind::StrLit) fail("type_mismatch", "VECTOR column needs a bracketed literal");
    v.kind = 2;
    v.vp = nullptr;
    v.vn = 0;
    parse_vector_into(e->sval, cd.len, v.v);
  }

  mj::Ptr run(const sql::Statement& st) {
    flush_wbuf();  // previous statement's bytes must be in the file before reads
    switch (st.kind) {
      case sql::StmtKind::Begin: {
        if (in_txn_) fail("unsupported_sql", "nested transactions are not supported");
        begin_txn();
        return ok_result();
      }
      case sql::StmtKind::Commit: {
        if (!in_txn_) return ok_result();
        commit();
        return ok_result();
      }
      case sql::StmtKind::Rollback: {
        if (!in_txn_) return ok_result();
        rollback_to(0);
        abort_txn();
        return ok_result();
      }
      case sql::StmtKind::CreateTable: return create_table(st);
      case sql::StmtKind::CreateVectorIndex: return create_index(st);
      case sql::StmtKind::Insert: return insert(st);
      case sql::StmtKind::Select: return select(st);
      case sql::StmtKind::Update: return update(st);
      case sql::StmtKind::Delete: return remove(st);
    }
    fail("unsupported_sql", "statement not supported");
  }

  mj::Ptr create_table(const sql::Statement& st) {
    if (tables_.count(st.table)) fail("table_already_exists", "table exists: " + st.table);
    auto tb = std::make_shared<Table>();
    tb->name = st.table;
    tb->cols = st.columns;
    finish_table(tb);
    save_catalog();
    return ok_result();
  }

  mj::Ptr create_index(const sql::Statement& st) {
    Table& tb = need_table(st.table);
    int pos = tb.col_pos(st.index_column);
    if (pos < 0) fail("column_not_found", "unknown column: " + st.index_column);
    if (tb.cols[size_t(pos)].type != sql::ColType::Vector) fail("type_mismatch", "index needs a vector column");
    tb.has_vector_index = true;
    tb.index_name = st.index_name;
    save_catalog();
    return ok_result();
  }

  mj::Ptr insert(const sql::Statement& st) {
    Table& tb = need_table(st.table);
    std::vector<int> order;
    if (st.insert_columns.empty()) {
      for (size_t k = 0; k < tb.cols.size(); ++k) order.push_back(int(k));
    } else {
      for (const auto& n : st.insert_columns) {
        int p = tb.col_pos(n);
        if (p < 0) fail("column_not_found", "unknown column: " + n);
        order.push_back(p);
      }
    }
    int64_t count = 0;
    ins_row_.resize(tb.cols.size());
    for (const auto& r : st.rows) {
      if (r.size() != order.size()) fail("column_count_mismatch", "VALUES count does not match columns");
      // Reset tags only: s/v keep their capacity across rows. A missing BIGINT
      // stays kind=0/i=0 (legacy default), missing VARCHAR/VECTOR still fail.
      for (auto& v : ins_row_) { v.kind = 0; v.i = 0; v.vp = nullptr; v.vn = 0; }
      for (size_t k = 0; k < r.size(); ++k) convert(tb.cols[size_t(order[k])], r[k], ins_row_[size_t(order[k])]);
      for (size_t k = 0; k < tb.cols.size(); ++k) {
        const sql::ColumnDef& cd = tb.cols[k];
        if (cd.type == sql::ColType::BigInt && ins_row_[k].kind != 0) fail("type_mismatch", "missing BIGINT value");
        if (cd.type == sql::ColType::VarChar && ins_row_[k].kind != 1) fail("type_mismatch", "missing VARCHAR value");
        if (cd.type == sql::ColType::Vector && ins_row_[k].kind != 2) fail("type_mismatch", "missing VECTOR value");
      }
      int64_t pk = ins_row_[size_t(tb.pk_index)].i;
      if (tb.index.contains(pk)) fail("duplicate_key", "duplicate primary key");
      const std::string& payload = encode_row(tb, ins_row_);
      insert_new_record(tb, payload, pk);
      ++count;
    }
    return affected(count);
  }

  mj::Ptr update(const sql::Statement& st) {
    Table& tb = need_table(st.table);
    for (const auto& kv : st.sets) {
      int p = tb.col_pos(kv.first);
      if (p < 0) fail("column_not_found", "unknown column: " + kv.first);
      if (p == tb.pk_index) fail("unsupported_sql", "primary key updates are not supported");
    }
    std::vector<int64_t> hits;
    tb.index.for_each_while([&](int64_t pk, size_t roff) -> bool {
      const Row& row = read_row(tb, roff);
      Ctx ctx{&tb, &row};
      if (test(st.where, ctx)) hits.push_back(pk);
      return true;
    });
    int64_t count = 0;
    for (int64_t pk : hits) {
      size_t off = tb.index.at(pk);
      Row row = read_row(tb, off);
      materialize_row(row);  // the copy must own its vectors before re-encoding
      for (const auto& kv : st.sets) {
        int p = tb.col_pos(kv.first);
        convert(tb.cols[size_t(p)], kv.second, row[size_t(p)]);
      }
      const std::string& payload = encode_row(tb, row);
      write_record(tb, 1, payload, pk, true, off);
      ++count;
    }
    return affected(count);
  }

  mj::Ptr remove(const sql::Statement& st) {
    Table& tb = need_table(st.table);
    std::vector<int64_t> hits;
    tb.index.for_each_while([&](int64_t pk, size_t roff) -> bool {
      const Row& row = read_row(tb, roff);
      Ctx ctx{&tb, &row};
      if (test(st.where, ctx)) hits.push_back(pk);
      return true;
    });
    for (int64_t pk : hits) {
      size_t off = tb.index.at(pk);
      std::string payload;
      put_u64(payload, uint64_t(pk));
      write_record(tb, 2, payload, pk, true, off);
    }
    return affected(int64_t(hits.size()));
  }

  mj::Ptr select(const sql::Statement& st) {
    Table& tb = need_table(st.table);
    // resolve projection
    std::vector<std::string> names;
    std::vector<sql::ExprPtr> exprs;
    for (const auto& it : st.items) {
      if (it.star) {
        for (const auto& c : tb.cols) {
          auto e = std::make_shared<sql::Expr>();
          e->kind = sql::ExprKind::Column;
          e->sval = c.name;
          names.push_back(c.name);
          exprs.push_back(e);
        }
      } else {
        std::string nm = it.alias;
        if (nm.empty()) {
          if (it.expr->kind == sql::ExprKind::Column) nm = it.expr->sval;
          else nm = "expr";
        }
        names.push_back(nm);
        exprs.push_back(it.expr);
      }
    }
    // order keys -> projection index or plain column
    struct Key { int proj; int col; bool desc; };
    std::vector<Key> keys;
    for (const auto& ok : st.order) {
      Key k{-1, -1, ok.desc};
      for (size_t j = 0; j < names.size(); ++j) if (names[j] == ok.name) { k.proj = int(j); break; }
      if (k.proj < 0) {
        k.col = tb.col_pos(ok.name);
        if (k.col < 0) fail("column_not_found", "unknown ORDER BY key: " + ok.name);
      }
      keys.push_back(k);
    }

    // ---- hot path: ORDER BY a single L2_DISTANCE + LIMIT ----
    // The generic loop builds Values and evaluates expressions per row. For the
    // benchmark shape (vector TopK with an optional scalar equality filter) the
    // distance can be computed straight from the cached record bytes, so the
    // scan does no allocation and no expression dispatch at all.
    if (keys.size() == 1 && st.has_limit && st.limit > 0 && keys[0].proj >= 0 &&
        exprs[size_t(keys[0].proj)]->kind == sql::ExprKind::L2) {
      const sql::ExprPtr le = exprs[size_t(keys[0].proj)];
      int vpos = le->lhs->kind == sql::ExprKind::Column ? tb.col_pos(le->lhs->sval) : -1;
      if (vpos >= 0 && tb.cols[size_t(vpos)].type == sql::ColType::Vector &&
          le->rhs->kind == sql::ExprKind::StrLit) {
        const int dim = tb.cols[size_t(vpos)].len;
        // Parse once: seed the per-statement cache so eval() below reuses it.
        // (unordered_map references stay valid across later emplaces/rehash.)
        const std::vector<float>& qv =
            l2_cache_.emplace(le.get(), parse_vector(le->rhs->sval, dim)).first->second;
        struct Pred { int col; sql::CmpOp op; int kind; int64_t iv; std::string sv; };
        std::vector<Pred> preds;
        preds.reserve(4);
        bool flat_ok = true;
        // self-recursive lambda: std::function would heap-allocate per query
        auto flat = [&](const sql::ExprPtr& e, auto&& self) -> void {
          if (!flat_ok || !e) return;
          if (e->kind == sql::ExprKind::And) { self(e->lhs, self); self(e->rhs, self); return; }
          if (e->kind != sql::ExprKind::Cmp || e->lhs->kind != sql::ExprKind::Column ||
              (e->rhs->kind != sql::ExprKind::IntLit && e->rhs->kind != sql::ExprKind::StrLit)) {
            flat_ok = false; return;
          }
          int c = tb.col_pos(e->lhs->sval);
          if (c < 0) { flat_ok = false; return; }
          // Literal kind must match the column type. The generic evaluator
          // coerces numeric strings against BIGINT; the fast paths do not, so
          // any mismatch (or a vector-column predicate) falls back to it.
          const sql::ColType ct = tb.cols[size_t(c)].type;
          if (ct == sql::ColType::Vector ||
              (ct == sql::ColType::BigInt && e->rhs->kind != sql::ExprKind::IntLit) ||
              (ct == sql::ColType::VarChar && e->rhs->kind != sql::ExprKind::StrLit)) {
            flat_ok = false; return;
          }
          Pred pr;
          pr.col = c; pr.op = e->cmp;
          pr.kind = e->rhs->kind == sql::ExprKind::IntLit ? 0 : 1;
          pr.iv = e->rhs->ival; pr.sv = e->rhs->sval;
          preds.push_back(std::move(pr));
        };
        flat(st.where, flat);
        if (flat_ok) {
          // ---- column-cache scan: no record I/O, no decode, no allocation ----
          // Slots are in pk order, so equal distances keep the smaller pk by
          // never displacing an incumbent (strict better_than).
          ensure_cache(tb);
          if (tb.cache.valid) {
            const VecCache& c = tb.cache;
            const char* mbase = tb.map_base + c.vec_off;
            const bool desc = keys[0].desc;
            const size_t K = size_t(st.limit);
            fast_preds_.resize(preds.size());
            for (size_t i = 0; i < preds.size(); ++i)
              fast_preds_[i] = {preds[i].col, preds[i].op, preds[i].iv};
            scan_cached_topk(tb, fast_preds_, qv.data(), K, desc, search_mode_, fast_top_);
            auto res = ok_result();
            auto cols = mj::Value::array();
            for (const auto& nm : names) cols->arr.push_back(mj::Value::str(nm));
            res->set("columns", cols);
            auto out = mj::Value::array();
            scratch_.resize(tb.cols.size());
            for (const CacheCand& cd : fast_top_) {
              for (size_t k = 0; k < tb.cols.size(); ++k) {
                Value& v = scratch_[k];
                if (tb.cols[k].type == sql::ColType::BigInt) {
                  v.kind = 0; v.i = c.cols[k][cd.slot]; v.vp = nullptr; v.vn = 0;
                } else {
                  v.kind = 2; v.vp = reinterpret_cast<const float*>(mbase + c.offs[cd.slot]); v.vn = size_t(dim);
                }
              }
              Ctx ctx{&tb, &scratch_};
              auto r = mj::Value::array();
              for (const auto& e : exprs) {
                if (e == le) r->arr.push_back(mj::Value::real(cd.d));
                else r->arr.push_back(to_json(eval(e, ctx)));
              }
              out->arr.push_back(r);
            }
            res->set("rows", out);
            return res;
          }
          // Fixed-width layout (no VARCHAR): every column offset is constant,
          // so the per-row schema walk can be hoisted out of the scan.
          bool fixed_layout = true;
          std::vector<size_t> coff(tb.cols.size(), 0);
          {
            size_t o = 8;
            for (size_t k = 0; k < tb.cols.size(); ++k) {
              if (int(k) == tb.pk_index) { coff[k] = 0; continue; }
              coff[k] = o;
              if (tb.cols[k].type == sql::ColType::BigInt) o += 8;
              else if (tb.cols[k].type == sql::ColType::VarChar) { fixed_layout = false; break; }
              else o += size_t(tb.cols[k].len) * 4;
            }
          }
          const size_t vec_off = fixed_layout ? coff[size_t(vpos)] : 0;
          const bool desc = keys[0].desc;
          struct Cand { double d; int64_t pk; };
          // ascending by rank: top[0] is the best kept row, top.back() the worst
          auto better_than = [&](const Cand& a, const Cand& b) {
            if (a.d != b.d) return desc ? a.d > b.d : a.d < b.d;
            return a.pk < b.pk;
          };
          const size_t K = size_t(st.limit);
          std::vector<Cand> top;
          top.reserve(K + 1);
          const float* bp = qv.data();
          tb.index.for_each_while([&](int64_t pk, size_t roff) -> bool {
            uint32_t rlen;
            const char* rec = record_ptr(tb, roff, rlen);
            bool pass = true;
            for (const Pred& pr : preds) {
              const char* cp;
              if (fixed_layout) cp = rec + coff[size_t(pr.col)];
              else if (pr.col == tb.pk_index) cp = rec;
              else {
                const char* q = rec + 8;
                cp = nullptr;
                for (size_t k = 0; k < tb.cols.size(); ++k) {
                  if (int(k) == tb.pk_index) continue;
                  if (int(k) == pr.col) { cp = q; break; }
                  if (tb.cols[k].type == sql::ColType::BigInt) q += 8;
                  else if (tb.cols[k].type == sql::ColType::VarChar) q += 4 + get_u32(q);
                  else q += size_t(tb.cols[k].len) * 4;
                }
                if (!cp) { pass = false; break; }
              }
              int c;
              if (pr.kind == 0) {
                int64_t a = int64_t(get_u64(cp));
                c = a < pr.iv ? -1 : (a > pr.iv ? 1 : 0);
              } else {
                uint32_t n = get_u32(cp);
                std::string a(cp + 4, n);
                c = a.compare(pr.sv) < 0 ? -1 : (a.compare(pr.sv) > 0 ? 1 : 0);
              }
              bool t;
              switch (pr.op) {
                case sql::CmpOp::Eq: t = c == 0; break;
                case sql::CmpOp::Lt: t = c < 0; break;
                case sql::CmpOp::Le: t = c <= 0; break;
                case sql::CmpOp::Gt: t = c > 0; break;
                default: t = c >= 0; break;
              }
              if (!t) { pass = false; break; }
            }
            if (!pass) return true;
            const char* vp = nullptr;
            if (fixed_layout) vp = rec + vec_off;
            else {
              const char* q = rec + 8;
              for (size_t k = 0; k < tb.cols.size(); ++k) {
                if (int(k) == tb.pk_index) continue;
                if (int(k) == vpos) { vp = q; break; }
                if (tb.cols[k].type == sql::ColType::BigInt) q += 8;
                else if (tb.cols[k].type == sql::ColType::VarChar) q += 4 + get_u32(q);
                else q += size_t(tb.cols[k].len) * 4;
              }
            }
            if (!vp) return true;
            const float* ap = reinterpret_cast<const float*>(vp);
            double dist = l2_kernel(ap, bp, dim);
            Cand c{dist, pk};
            if (top.size() < K) {
              top.push_back(c);
              size_t j = top.size() - 1;
              while (j > 0 && better_than(top[j], top[j - 1])) { std::swap(top[j], top[j - 1]); --j; }
            } else if (better_than(c, top.back())) {
              top.back() = c;
              size_t j = K - 1;
              while (j > 0 && better_than(top[j], top[j - 1])) { std::swap(top[j], top[j - 1]); --j; }
            }
            return true;
          });
          auto res = ok_result();
          auto cols = mj::Value::array();
          for (const auto& nm : names) cols->arr.push_back(mj::Value::str(nm));
          res->set("columns", cols);
          auto out = mj::Value::array();
          for (const Cand& c : top) {
            const Row& row = read_row(tb, tb.index.at(c.pk));
            Ctx ctx{&tb, &row};
            auto r = mj::Value::array();
            for (const auto& e : exprs) r->arr.push_back(to_json(eval(e, ctx)));
            out->arr.push_back(r);
          }
          res->set("rows", out);
          return res;
        }
      }
    }

    // Only the projection and the ordering keys are retained: keeping whole rows
    // would copy every vector and blow the memory budget on a large table.
    struct Out {
      int64_t pk;
      std::vector<Value> vals;
      std::vector<Value> keyvals;
    };
    auto cmp_keys = [&](const Value* ka, int64_t pa, const Value* kb, int64_t pb) {
      for (size_t j = 0; j < keys.size(); ++j) {
        const Value& va = ka[j];
        const Value& vb = kb[j];
        int c;
        if (va.kind == 3 || vb.kind == 3) {
          double x = distance_of(va), y = distance_of(vb);
          c = x < y ? -1 : (x > y ? 1 : 0);
        } else if (va.kind == 0) c = va.i < vb.i ? -1 : (va.i > vb.i ? 1 : 0);
        else c = va.s.compare(vb.s) < 0 ? -1 : (va.s.compare(vb.s) > 0 ? 1 : 0);
        if (keys[j].desc) c = -c;
        if (c) return c < 0;
      }
      return pa < pb;  // ties always break on primary key ascending
    };
    auto better = [&](const Out& a, const Out& b) {
      return cmp_keys(a.keyvals.data(), a.pk, b.keyvals.data(), b.pk);
    };
    // push_heap puts the comp-greatest element at the front; using `better`
    // itself makes that front the WORST kept row, which is the one we evict.

    const bool bounded = st.has_limit;
    const size_t cap = bounded ? size_t(st.limit) : SIZE_MAX;
    std::vector<Out> rows;
    if (cap != 0) {
      tb.index.for_each_while([&](int64_t pk, size_t roff) -> bool {
        const Row& row = read_row(tb, roff);
        Ctx ctx{&tb, &row};
        if (!test(st.where, ctx)) return true;
        if (keys.empty()) {
          Out o;
          o.pk = pk;
          for (const auto& e : exprs) o.vals.push_back(eval(e, ctx));
          rows.push_back(std::move(o));
          if (bounded && rows.size() >= cap) return false;
          return true;
        }
        // Ordering keys first: a row that cannot enter the top-K is dropped
        // without evaluating the projection or allocating anything.
        std::array<Value, 4> kbuf;
        std::vector<Value> kbig;
        Value* kp;
        if (keys.size() <= kbuf.size()) kp = kbuf.data();
        else { kbig.resize(keys.size()); kp = kbig.data(); }
        for (size_t j = 0; j < keys.size(); ++j) {
          const Key& k = keys[j];
          if (k.proj >= 0) kp[j] = eval(exprs[size_t(k.proj)], ctx);
          else kp[j] = row[size_t(k.col)];
        }
        if (rows.size() == cap && !cmp_keys(kp, pk, rows.front().keyvals.data(), rows.front().pk))
          return true;
        Out o;
        o.pk = pk;
        for (const auto& e : exprs) o.vals.push_back(eval(e, ctx));
        o.keyvals.assign(kp, kp + keys.size());
        if (rows.size() < cap) {
          rows.push_back(std::move(o));
          std::push_heap(rows.begin(), rows.end(), better);
        } else if (better(o, rows.front())) {
          std::pop_heap(rows.begin(), rows.end(), better);
          rows.back() = std::move(o);
          std::push_heap(rows.begin(), rows.end(), better);
        }
        return true;
      });
    }
    if (!keys.empty()) std::sort(rows.begin(), rows.end(), better);

    size_t n = rows.size();
    auto res = ok_result();
    auto cols = mj::Value::array();
    for (const auto& nm : names) cols->arr.push_back(mj::Value::str(nm));
    res->set("columns", cols);
    auto out = mj::Value::array();
    for (size_t k = 0; k < n; ++k) {
      auto r = mj::Value::array();
      for (const auto& v : rows[k].vals) r->arr.push_back(to_json(v));
      out->arr.push_back(r);
    }
    res->set("rows", out);
    return res;
  }
};

}  // namespace db