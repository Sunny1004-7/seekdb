# MiniVectorDB

单机、单进程、单客户端的轻量向量数据库。支持基础 SQL、L2 向量检索、持久化存储、
显式事务与崩溃恢复。通过 **stdin/stdout 的 JSON Lines 协议**与测试驱动通信，
不监听任何网络端口。

- 语言 / 标准：**C++17**，仅使用标准库与 POSIX 文件接口（`pread`/`pwrite`/`fsync`）
- 第三方依赖：**无**。存储、查询执行、向量索引与事务核心均为自研
- 目标平台：Linux x86_64
- 数据目录：**`/data`**（唯一写入路径）

## 1. 构建与启动

### 镜像构建

```bash
docker build -t minivectordb:latest .
```

多阶段构建：`ubuntu:24.04` + `g++` 编译，产物静态链接 libstdc++/libgcc，
运行阶段只需 base 镜像自带的 libc/libm。依赖全部在构建阶段安装并锁定，
运行时不需要网络。

### 前台启动（保持 stdin 打开，不分配 TTY，不映射端口）

```bash
mkdir -p ./data && chmod 777 ./data
docker run --rm -i --network=none \
  --cpus=2 --memory=2g --memory-swap=2g --pids-limit=128 \
  --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --mount "type=bind,src=$(pwd)/data,dst=/data" \
  minivectordb:latest
```

也可以直接跑二进制（本地开发）：

```bash
g++ -O3 -std=c++17 -o build/mvdb src/main.cpp
./build/mvdb ./data          # argv[1] 为数据目录，缺省 /data
```

程序启动后先完成恢复，再主动输出 `{"type":"ready"}` 并 flush，然后才开始读请求。
日志与诊断信息一律写 stderr，stdout 只有协议报文。

## 2. 协议

请求 / 响应各占一个物理行，UTF-8，每行一个完整 JSON 对象，输出后立即 flush。

```jsonc
// stdin
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));","search_mode":"exact"}
// stdout
{"id":1,"ok":true}
```

- 查询成功：`{"id":N,"ok":true,"columns":[...],"rows":[[...],...]}`
- 修改成功：`{"id":N,"ok":true,"affected_rows":K}`
- 失败：`{"id":N,"ok":false,"error":{"code":"...","message":"..."}}`
- 关闭：`{"id":N,"op":"shutdown"}` → `{"id":N,"ok":true}`，随后以退出码 0 结束，
  关闭响应之后不再向 stdout 输出任何内容

`id` 原样回传并保持 JSON 类型；BIGINT 用 JSON 整数（不经浮点）；向量投影为 JSON 数组。

## 3. SQL 支持范围

| 类别 | 支持 |
|---|---|
| 表结构 | `CREATE TABLE`；`BIGINT`、`VARCHAR(n)`、`VECTOR(n)`；一个 BIGINT 主键；每表最多一个向量列 |
| 数据操作 | `INSERT`（含多行、可选列清单）、`SELECT`、`UPDATE`、`DELETE` |
| 条件 | `=`、`<`、`<=`、`>`、`>=`、`AND` |
| 投影 | 普通列、`SELECT *`、`L2_DISTANCE(col, '[...]') AS alias` |
| 排序 / 分页 | `ORDER BY <列或别名> [ASC\|DESC]`、`LIMIT n` |
| 向量索引 | `CREATE VECTOR INDEX name ON t(vec_col)` |
| 事务 | `BEGIN` / `COMMIT` / `ROLLBACK`；未显式 BEGIN 时按自动提交 |

语义要点：

- `VARCHAR(n)` 长度按 **Unicode 字符数**计；字符串比较按 **Unicode 码点顺序**
- 向量为固定维度 float32；拒绝 NaN / Infinity / 维度不符 / 转换溢出；
  距离按**存储后的 float32** 计算
- 带过滤的 TopK 在**满足 WHERE 的集合内**计算，不是先取 K 再过滤
- 距离并列、普通 `ORDER BY` 并列，一律按**主键升序**；未指定 `ORDER BY` 时结果按集合比较
- 结果不足 K 条时返回全部符合条件的记录；空集合返回空 `rows` 但保留 `columns`
- `exact` 与 `ann` 两种 `search_mode` 目前都返回**精确结果**（Recall@10 = 1.0）

**不支持**（返回 `unsupported_sql`，绝不静默改写成别的含义）：
`NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、外键、嵌套事务、主键更新、
`OR`、`!=`/`<>`、`LIKE`/`IN`/`BETWEEN`、`DISTINCT`、`GROUP BY`/`HAVING`、`LIMIT offset`。

### 错误码

| code | 触发 |
|---|---|
| `duplicate_key` | 主键重复 |
| `dimension_mismatch` | 向量维度与列定义不符 |
| `unsupported_sql` | 语句或子句超出上表范围 |
| `invalid_vector` | 向量字面量格式错误 / 含 NaN / Infinity |
| `value_out_of_range` | 整数或 float32 溢出 |
| `string_too_long` | VARCHAR 超过声明长度 |
| `type_mismatch` | 值与列类型不兼容 |
| `column_count_mismatch` | VALUES 个数与列数不符 |
| `table_not_found` / `table_already_exists` / `column_not_found` | 对象不存在 / 已存在 |
| `syntax_error` | 文本不合法 |
| `io_error` / `internal_error` | 落盘失败 / 内部异常 |

预期中的 SQL 错误不会终止进程：返回失败响应后继续接收下一条请求。

## 4. 从空数据目录开始的完整示例

```jsonc
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, cat VARCHAR(8), v VECTOR(2));","search_mode":"exact"}
{"id":2,"op":"execute","sql":"INSERT INTO docs VALUES (1,'a','[3,4]'),(2,'b','[0,1]');","search_mode":"exact"}
{"id":3,"op":"execute","sql":"SELECT id, L2_DISTANCE(v,'[0,0]') AS d FROM docs ORDER BY d LIMIT 1;","search_mode":"exact"}
// -> {"id":3,"ok":true,"columns":["id","d"],"rows":[[2,1.0]]}
{"id":4,"op":"execute","sql":"BEGIN;","search_mode":"exact"}
{"id":5,"op":"execute","sql":"UPDATE docs SET cat='z' WHERE id=1;","search_mode":"exact"}
{"id":6,"op":"execute","sql":"ROLLBACK;","search_mode":"exact"}
{"id":7,"op":"execute","sql":"SELECT id, cat FROM docs ORDER BY id;","search_mode":"exact"}
// -> rows [[1,"a"],[2,"b"]]
{"id":8,"op":"shutdown"}
```

重启（复用同一 `/data`）后再次发送第 7 条，结果不变。

## 5. 存储、事务与恢复设计

- **记录文件 `<table>.rec`**：追加式的版本化记录。每条
  `[u32 len][u64 txn][u8 op][payload][u32 crc32]`，`op=1` 为整行镜像、`op=2` 为删除标记。
- **提交日志 `commits.log`**：每条 `[u64 txn][u32 crc32]`。
  一个事务只有在它的 id 落进这个文件后才可见。
- **提交顺序**：先 `fsync` 本次事务碰过的记录文件，再追加提交标记并 `fsync`。
  因此「已回 COMMIT 成功」的事务一定可恢复；提交请求发出但未收到响应的，
  恢复后要么全生效要么全不生效。
- **目录 `catalog.json`**：临时文件 + `fsync` + `rename` 原子替换。
- **内存**：只常驻 `主键 -> 记录偏移` 索引；记录体经 **8 MiB 有界块缓存**读取，
  向量以**零拷贝视图**参与距离计算。因此数据量可以远超内存预算。
- **事务**：同一时刻最多一个显式事务。事务内读到自己的修改。
  语句级失败时回滚该语句；若事务仍将继续，会追加**补偿记录**，
  避免后续 COMMIT 把已中止的行恢复出来。
- **恢复**：扫描记录文件，只重放 `commits.log` 里的事务。
  `next_txn` 取「已提交 id」与「记录中出现过的最大 id（含未提交）」的最大值 +1，
  **保证崩溃后事务 id 不被复用**。
- **正常关闭**：回滚未提交事务、落盘、回响应、退出码 0。
- **向量索引**：`CREATE VECTOR INDEX` 会被持久化；当前 `ann` 模式仍走精确扫描，
  召回率恒为 1.0，索引维护（插入/更新/删除后立即反映）由记录版本化天然保证。

### 已知限制

- 记录文件随更新增长，尚未实现空间回收（compaction）。
- `ann` 模式尚未接入近似索引，与 `exact` 同路径。
- 无 `ORDER BY` 且无 `LIMIT` 的全表投影会物化全部命中行。

## 6. 本地自测

```bash
# 官方题包的协议检查（需在 Linux/WSL 下运行，run.py 使用 POSIX 专有 API）
cd 题包
python3 protocol_check/run.py --command '["/abs/path/build/mvdb","{data_dir}"]' --output ../out

# 崩溃注入（kill -9 中途、重复恢复、语句中止后提交、回滚后崩溃）
python3 tests/crash_test.py ./build/mvdb

# 官方口径性能基准 + 暴力真值校验
./build/gen_truth 10000 128 500 /tmp/b.sql /tmp/bt.json
python3 tests/bench.py ./build/mvdb /tmp/b.sql /tmp/bt.json

# 超内存场景（数据远超 ulimit -v）
./build/gen_truth 200000 128 12 /tmp/big.sql /tmp/big.json
python3 tests/big_table_test.py ./build/mvdb /tmp/big.sql /tmp/big.json 64
```
