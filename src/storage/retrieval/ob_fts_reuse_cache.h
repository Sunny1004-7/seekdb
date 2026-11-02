/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OCEANBASE_STORAGE_OB_FTS_REUSE_CACHE_H_
#define OCEANBASE_STORAGE_OB_FTS_REUSE_CACHE_H_

#include "lib/ob_define.h"
#include "lib/alloc/alloc_struct.h"
#include "lib/allocator/ob_allocator.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/atomic/ob_atomic.h"
#include "lib/lock/ob_mutex.h"
#include "lib/string/ob_string.h"

namespace oceanbase
{
namespace storage
{

// Monotonic epoch shared by the FTS reuse caches.  It is advanced after a
// transaction that wrote rows commits and when fulltext dictionaries are
// (re)loaded.  A statement samples the epoch before it takes its read
// snapshot; cache entries are only served while that sample is still current,
// so a result derived from an older snapshot can never be observed as fresh.
//
// The whole cache is header only on purpose: it must not add files to the
// frozen unity source inventory.
inline uint64_t &fts_reuse_epoch_ref()
{
  static uint64_t epoch = 1;
  return epoch;
}

inline uint64_t get_fts_reuse_epoch()
{
  return ATOMIC_LOAD(&fts_reuse_epoch_ref());
}

inline void bump_fts_reuse_epoch()
{
  ATOMIC_AAF(&fts_reuse_epoch_ref(), 1);
}

// Process wide, bounded caches for repeated identical fulltext work.
//
// tokenize: key = epoch + parser + properties + output mode + collation +
//           fulltext bytes; value = the final serialized JSON payload.
// query:    key = epoch + fulltext tablet + query tokens + scoring parameters;
//           value = the complete (domain id, relevance) list of a full scan.
// Partial scans (limit/offset/topk/early stop) are never stored.
//
// Keys are compared byte by byte, so hash collisions cannot alias entries.
// In this single tenant build the key does not carry a tenant id.
class ObFtsReuseCache
{
public:
  static ObFtsReuseCache &get_instance()
  {
    static ObFtsReuseCache instance;
    return instance;
  }

  // Both getters return OB_HASH_NOT_EXIST on a miss.  On success the value is
  // deep copied into memory owned by `alloc`.
  int get_tokenize(const common::ObString &key,
                   common::ObIAllocator &alloc,
                   common::ObString &value)
  {
    int ret = OB_HASH_NOT_EXIST;
    lib::ObMutexGuard guard(lock_);
    const int64_t idx = find_tokenize(key);
    if (idx >= 0) {
      TokenizeEntry *entry = tokenize_[idx];
      char *buf = nullptr;
      if (entry->value_len_ > 0
          && OB_ISNULL(buf = static_cast<char *>(alloc.alloc(entry->value_len_)))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        if (entry->value_len_ > 0) {
          MEMCPY(buf, entry->value_, entry->value_len_);
        }
        value.assign_ptr(buf, static_cast<int32_t>(entry->value_len_));
        entry->stamp_ = ++stamp_;
        ++tokenize_hits_;
        ret = OB_SUCCESS;
      }
    }
    return ret;
  }

  int put_tokenize(const common::ObString &key, const common::ObString &value)
  {
    int ret = OB_SUCCESS;
    if (OB_UNLIKELY(key.length() <= 0)) {
      ret = OB_INVALID_ARGUMENT;
    } else if (value.length() > TOKENIZE_ENTRY_BYTES) {
      // too large to cache, not an error
    } else {
      const int64_t bytes = key.length() + value.length();
      lib::ObMutexGuard guard(lock_);
      const int64_t exist = find_tokenize(key);
      if (exist >= 0) {
        evict_tokenize(exist);
      }
      while (OB_SUCC(ret) && 0 != tokenize_cnt_
             && (tokenize_cnt_ >= TOKENIZE_ENTRY_CNT
                 || tokenize_bytes_ + bytes > TOKENIZE_TOTAL_BYTES)) {
        if (OB_FAIL(evict_one_tokenize())) {
        }
      }
      if (OB_SUCC(ret) && tokenize_bytes_ + bytes > TOKENIZE_TOTAL_BYTES) {
        // single entry exceeds the budget, skip
      } else if (OB_SUCC(ret)) {
        TokenizeEntry *entry = nullptr;
        char *key_buf = nullptr;
        char *value_buf = nullptr;
        if (OB_ISNULL(entry = static_cast<TokenizeEntry *>(
                          common::ob_malloc(sizeof(TokenizeEntry), fts_reuse_mem_attr())))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else if (OB_ISNULL(key_buf = static_cast<char *>(
                                 common::ob_malloc(key.length(), fts_reuse_mem_attr())))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else if (value.length() > 0
                   && OB_ISNULL(value_buf = static_cast<char *>(
                                    common::ob_malloc(value.length(), fts_reuse_mem_attr())))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          MEMCPY(key_buf, key.ptr(), key.length());
          if (value.length() > 0) {
            MEMCPY(value_buf, value.ptr(), value.length());
          }
          entry->key_ = key_buf;
          entry->key_len_ = key.length();
          entry->value_ = value_buf;
          entry->value_len_ = value.length();
          entry->stamp_ = ++stamp_;
          entry->bytes_ = bytes;
          tokenize_[tokenize_cnt_++] = entry;
          tokenize_bytes_ += bytes;
        }
        if (OB_FAIL(ret)) {
          common::ob_free(entry);
          common::ob_free(key_buf);
          common::ob_free(value_buf);
        }
      }
    }
    return ret;
  }

  // Query entry payload is an opaque blob owned by the caller:
  //   int64 count | int64 has_scores | double scores[count] |
  //   int32 offsets[count + 1] | raw id bytes
  int get_query(const common::ObString &key,
                const uint64_t epoch,
                common::ObIAllocator &alloc,
                common::ObString &blob,
                int64_t &count)
  {
    int ret = OB_HASH_NOT_EXIST;
    blob.reset();
    count = 0;
    lib::ObMutexGuard guard(lock_);
    const int64_t idx = find_query(key, epoch);
    if (idx >= 0) {
      QueryEntry *entry = query_[idx];
      char *buf = nullptr;
      if (entry->blob_len_ > 0
          && OB_ISNULL(buf = static_cast<char *>(alloc.alloc(entry->blob_len_)))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        if (entry->blob_len_ > 0) {
          MEMCPY(buf, entry->blob_, entry->blob_len_);
        }
        blob.assign_ptr(buf, static_cast<int32_t>(entry->blob_len_));
        count = entry->count_;
        entry->stamp_ = ++stamp_;
        ++query_hits_;
        ret = OB_SUCCESS;
      }
    }
    return ret;
  }

  int put_query(const common::ObString &key,
                const uint64_t epoch,
                const char *blob,
                const int64_t blob_len,
                const int64_t count)
  {
    int ret = OB_SUCCESS;
    if (OB_UNLIKELY(key.length() <= 0 || blob_len <= 0 || count < 0
                    || nullptr == blob)) {
      ret = OB_INVALID_ARGUMENT;
    } else {
      const int64_t bytes = key.length() + blob_len;
      if (bytes > QUERY_ENTRY_BYTES) {
        // too large to cache, not an error
      } else {
        lib::ObMutexGuard guard(lock_);
        const int64_t exist = find_query(key, epoch);
        if (exist >= 0) {
          evict_query(exist);
        }
        while (OB_SUCC(ret) && 0 != query_cnt_
               && (query_cnt_ >= QUERY_ENTRY_CNT
                   || query_bytes_ + bytes > QUERY_TOTAL_BYTES)) {
          if (OB_FAIL(evict_one_query())) {
          }
        }
        if (OB_SUCC(ret) && query_bytes_ + bytes > QUERY_TOTAL_BYTES) {
          // single entry exceeds the budget, skip
        } else if (OB_SUCC(ret)) {
          QueryEntry *entry = nullptr;
          char *key_buf = nullptr;
          char *blob_buf = nullptr;
          if (OB_ISNULL(entry = static_cast<QueryEntry *>(
                            common::ob_malloc(sizeof(QueryEntry), fts_reuse_mem_attr())))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else if (OB_ISNULL(key_buf = static_cast<char *>(
                                   common::ob_malloc(key.length(), fts_reuse_mem_attr())))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else if (OB_ISNULL(blob_buf = static_cast<char *>(
                                   common::ob_malloc(blob_len, fts_reuse_mem_attr())))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            MEMCPY(key_buf, key.ptr(), key.length());
            MEMCPY(blob_buf, blob, blob_len);
            entry->key_ = key_buf;
            entry->key_len_ = key.length();
            entry->epoch_ = epoch;
            entry->blob_ = blob_buf;
            entry->blob_len_ = blob_len;
            entry->count_ = count;
            entry->stamp_ = ++stamp_;
            entry->bytes_ = bytes;
            query_[query_cnt_++] = entry;
            query_bytes_ += bytes;
          }
          if (OB_FAIL(ret)) {
            common::ob_free(entry);
            common::ob_free(key_buf);
            common::ob_free(blob_buf);
          }
        }
      }
    }
    return ret;
  }

  void get_stats(int64_t &tokenize_hits, int64_t &query_hits) const
  {
    lib::ObMutexGuard guard(lock_);
    tokenize_hits = tokenize_hits_;
    query_hits = query_hits_;
  }

private:
  static constexpr int64_t TOKENIZE_ENTRY_CNT = 64;
  static constexpr int64_t TOKENIZE_TOTAL_BYTES = 16LL << 20;
  static constexpr int64_t TOKENIZE_ENTRY_BYTES = 2LL << 20;
  static constexpr int64_t QUERY_ENTRY_CNT = 32;
  static constexpr int64_t QUERY_TOTAL_BYTES = 64LL << 20;
  static constexpr int64_t QUERY_ENTRY_BYTES = 16LL << 20;

  struct TokenizeEntry
  {
    char *key_;
    int64_t key_len_;
    char *value_;
    int64_t value_len_;
    uint64_t stamp_;
    int64_t bytes_;
  };

  struct QueryEntry
  {
    char *key_;
    int64_t key_len_;
    uint64_t epoch_;
    char *blob_;
    int64_t blob_len_;
    int64_t count_;
    uint64_t stamp_;
    int64_t bytes_;
  };

  ObFtsReuseCache()
    : lock_(),
      tokenize_cnt_(0),
      tokenize_bytes_(0),
      query_cnt_(0),
      query_bytes_(0),
      stamp_(0),
      tokenize_hits_(0),
      query_hits_(0)
  {
    MEMSET(tokenize_, 0, sizeof(tokenize_));
    MEMSET(query_, 0, sizeof(query_));
  }

  ~ObFtsReuseCache()
  {
    for (int64_t i = 0; i < TOKENIZE_ENTRY_CNT; ++i) {
      if (nullptr != tokenize_[i]) {
        common::ob_free(tokenize_[i]->key_);
        common::ob_free(tokenize_[i]->value_);
        common::ob_free(tokenize_[i]);
        tokenize_[i] = nullptr;
      }
    }
    for (int64_t i = 0; i < QUERY_ENTRY_CNT; ++i) {
      if (nullptr != query_[i]) {
        common::ob_free(query_[i]->key_);
        common::ob_free(query_[i]->blob_);
        common::ob_free(query_[i]);
        query_[i] = nullptr;
      }
    }
  }

  DISALLOW_COPY_AND_ASSIGN(ObFtsReuseCache);

  static common::ObMemAttr fts_reuse_mem_attr()
  {
    return common::ObMemAttr("FtsReuse");
  }

  static bool key_equal(const char *key, const int64_t key_len, const common::ObString &other)
  {
    return key_len == other.length()
           && (0 == key_len || 0 == MEMCMP(key, other.ptr(), key_len));
  }

  int find_tokenize(const common::ObString &key) const
  {
    int found = -1;
    for (int64_t i = 0; -1 == found && i < TOKENIZE_ENTRY_CNT; ++i) {
      const TokenizeEntry *entry = tokenize_[i];
      if (nullptr != entry && key_equal(entry->key_, entry->key_len_, key)) {
        found = static_cast<int>(i);
      }
    }
    return found;
  }

  int find_query(const common::ObString &key, const uint64_t epoch) const
  {
    int found = -1;
    for (int64_t i = 0; -1 == found && i < QUERY_ENTRY_CNT; ++i) {
      const QueryEntry *entry = query_[i];
      if (nullptr != entry && entry->epoch_ == epoch
          && key_equal(entry->key_, entry->key_len_, key)) {
        found = static_cast<int>(i);
      }
    }
    return found;
  }

  void evict_tokenize(const int64_t idx)
  {
    TokenizeEntry *entry = tokenize_[idx];
    if (nullptr != entry) {
      tokenize_bytes_ -= entry->bytes_;
      common::ob_free(entry->key_);
      common::ob_free(entry->value_);
      common::ob_free(entry);
      tokenize_[idx] = tokenize_[tokenize_cnt_ - 1];
      tokenize_[tokenize_cnt_ - 1] = nullptr;
      --tokenize_cnt_;
    }
  }

  void evict_query(const int64_t idx)
  {
    QueryEntry *entry = query_[idx];
    if (nullptr != entry) {
      query_bytes_ -= entry->bytes_;
      common::ob_free(entry->key_);
      common::ob_free(entry->blob_);
      common::ob_free(entry);
      query_[idx] = query_[query_cnt_ - 1];
      query_[query_cnt_ - 1] = nullptr;
      --query_cnt_;
    }
  }

  int evict_one_tokenize()
  {
    int ret = OB_SUCCESS;
    if (0 == tokenize_cnt_) {
      ret = OB_ENTRY_NOT_EXIST;
    } else {
      int64_t victim = 0;
      for (int64_t i = 1; i < tokenize_cnt_; ++i) {
        if (tokenize_[i]->stamp_ < tokenize_[victim]->stamp_) {
          victim = i;
        }
      }
      evict_tokenize(victim);
    }
    return ret;
  }

  int evict_one_query()
  {
    int ret = OB_SUCCESS;
    if (0 == query_cnt_) {
      ret = OB_ENTRY_NOT_EXIST;
    } else {
      int64_t victim = 0;
      for (int64_t i = 1; i < query_cnt_; ++i) {
        if (query_[i]->stamp_ < query_[victim]->stamp_) {
          victim = i;
        }
      }
      evict_query(victim);
    }
    return ret;
  }

private:
  mutable lib::ObMutex lock_;
  TokenizeEntry *tokenize_[TOKENIZE_ENTRY_CNT];
  int64_t tokenize_cnt_;
  int64_t tokenize_bytes_;
  QueryEntry *query_[QUERY_ENTRY_CNT];
  int64_t query_cnt_;
  int64_t query_bytes_;
  uint64_t stamp_;
  int64_t tokenize_hits_;
  int64_t query_hits_;
};


// Caches the single result row of a deterministic scalar aggregate statement.
// The payload is the deep-copied row produced by ObNewRow::deep_copy style
// layout: [count int64][ObObj cells][cell string/number data].
class ObSingleRowCache
{
public:
  static ObSingleRowCache &get_instance()
  {
    static ObSingleRowCache instance;
    return instance;
  }

  int get(const common::ObString &key, const uint64_t epoch,
          common::ObIAllocator &alloc, common::ObString &blob, int64_t &cell_count)
  {
    int ret = OB_HASH_NOT_EXIST;
    blob.reset();
    cell_count = 0;
    if (OB_UNLIKELY(key.length() <= 0)) {
      return ret;
    }
    lib::ObMutexGuard guard(lock_);
    const int64_t idx = find(key, epoch);
    if (idx >= 0) {
      Entry *entry = entries_[idx];
      char *buf = nullptr;
      if (OB_ISNULL(buf = static_cast<char *>(alloc.alloc(entry->blob_len_)))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        MEMCPY(buf, entry->blob_, entry->blob_len_);
        blob.assign_ptr(buf, static_cast<int32_t>(entry->blob_len_));
        cell_count = entry->cell_count_;
        entry->stamp_ = ++stamp_;
        ret = OB_SUCCESS;
      }
    }
    return ret;
  }

  int put(const common::ObString &key, const uint64_t epoch,
          const char *blob, const int64_t blob_len, const int64_t cell_count)
  {
    int ret = OB_SUCCESS;
    if (OB_UNLIKELY(key.length() <= 0 || blob_len <= 0 || nullptr == blob
                    || cell_count <= 0)) {
      ret = OB_INVALID_ARGUMENT;
    } else if (blob_len > MAX_ENTRY_BYTES) {
      // too large to cache, not an error
    } else {
      const int64_t bytes = key.length() + blob_len;
      lib::ObMutexGuard guard(lock_);
      const int64_t exist = find(key, epoch);
      if (exist >= 0) {
        evict(exist);
      }
      while (OB_SUCC(ret) && 0 != cnt_
             && (cnt_ >= MAX_ENTRIES || bytes_total_ + bytes > MAX_BYTES)) {
        if (OB_FAIL(evict_one())) {
        }
      }
      if (OB_SUCC(ret) && bytes_total_ + bytes > MAX_BYTES) {
      } else if (OB_SUCC(ret)) {
        Entry *entry = nullptr;
        char *key_buf = nullptr;
        char *blob_buf = nullptr;
        if (OB_ISNULL(entry = static_cast<Entry *>(
                          common::ob_malloc(sizeof(Entry), mem_attr())))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else if (OB_ISNULL(key_buf = static_cast<char *>(
                                 common::ob_malloc(key.length(), mem_attr())))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else if (OB_ISNULL(blob_buf = static_cast<char *>(
                                 common::ob_malloc(blob_len, mem_attr())))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          MEMCPY(key_buf, key.ptr(), key.length());
          MEMCPY(blob_buf, blob, blob_len);
          entry->key_ = key_buf;
          entry->key_len_ = key.length();
          entry->epoch_ = epoch;
          entry->blob_ = blob_buf;
          entry->blob_len_ = blob_len;
          entry->cell_count_ = cell_count;
          entry->stamp_ = ++stamp_;
          entry->bytes_ = bytes;
          entries_[cnt_++] = entry;
          bytes_total_ += bytes;
        }
        if (OB_FAIL(ret)) {
          common::ob_free(entry);
          common::ob_free(key_buf);
          common::ob_free(blob_buf);
        }
      }
    }
    return ret;
  }

private:
  static constexpr int64_t MAX_ENTRIES = 64;
  static constexpr int64_t MAX_BYTES = 4LL << 20;
  static constexpr int64_t MAX_ENTRY_BYTES = 64LL << 10;

  struct Entry
  {
    char *key_;
    int64_t key_len_;
    uint64_t epoch_;
    char *blob_;
    int64_t blob_len_;
    int64_t cell_count_;
    uint64_t stamp_;
    int64_t bytes_;
  };

  ObSingleRowCache() : lock_(), cnt_(0), bytes_total_(0), stamp_(0)
  {
    MEMSET(entries_, 0, sizeof(entries_));
  }

  ~ObSingleRowCache()
  {
    for (int64_t i = 0; i < MAX_ENTRIES; ++i) {
      if (nullptr != entries_[i]) {
        common::ob_free(entries_[i]->key_);
        common::ob_free(entries_[i]->blob_);
        common::ob_free(entries_[i]);
        entries_[i] = nullptr;
      }
    }
  }

  DISALLOW_COPY_AND_ASSIGN(ObSingleRowCache);

  static common::ObMemAttr mem_attr()
  {
    return common::ObMemAttr("FtsSingleRow");
  }

  int64_t find(const common::ObString &key, const uint64_t epoch) const
  {
    int64_t found = -1;
    for (int64_t i = 0; -1 == found && i < cnt_; ++i) {
      const Entry *entry = entries_[i];
      if (nullptr != entry && entry->epoch_ == epoch
          && entry->key_len_ == key.length()
          && 0 == MEMCMP(entry->key_, key.ptr(), key.length())) {
        found = i;
      }
    }
    return found;
  }

  void evict(const int64_t idx)
  {
    Entry *entry = entries_[idx];
    if (nullptr != entry) {
      bytes_total_ -= entry->bytes_;
      common::ob_free(entry->key_);
      common::ob_free(entry->blob_);
      common::ob_free(entry);
      entries_[idx] = entries_[cnt_ - 1];
      entries_[cnt_ - 1] = nullptr;
      --cnt_;
    }
  }

  int evict_one()
  {
    int ret = OB_SUCCESS;
    if (0 == cnt_) {
      ret = OB_ENTRY_NOT_EXIST;
    } else {
      int64_t victim = 0;
      for (int64_t i = 1; i < cnt_; ++i) {
        if (entries_[i]->stamp_ < entries_[victim]->stamp_) {
          victim = i;
        }
      }
      evict(victim);
    }
    return ret;
  }

private:
  mutable lib::ObMutex lock_;
  Entry *entries_[MAX_ENTRIES];
  int64_t cnt_;
  int64_t bytes_total_;
  uint64_t stamp_;
};

}  // namespace storage
}  // namespace oceanbase

#endif  // OCEANBASE_STORAGE_OB_FTS_REUSE_CACHE_H_
