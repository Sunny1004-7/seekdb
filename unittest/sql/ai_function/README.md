# AI function native regression checks

These independently authored checks run SQL against a locally compiled seekdb.
The protocol fixture is not the organizer's Mock AI Service and does not require
models, credentials, or Internet access. It returns text-dependent vectors and
reverses OpenAI embedding indices to detect row-identity errors. Rerank fixtures
include cross-batch ties and duplicate text with distinct IDs. Authorization
values are never logged. All endpoint credentials used here are synthetic.

Run the fixture in a separate terminal (Python 3.10+):

```sh
python3 protocol_fixture.py --log /tmp/ai-protocol-requests.jsonl
```

Start a seekdb binary built from the source under test on port 2883, create the
`test` database, and ensure `pymysql` is available in the test client's Python.
Use a disposable test instance: the acceptance script creates `r10_*` models,
endpoints and tables, and temporarily sets `_rowsets_max_rows` to 16 before
restoring 256. The fixture binds only 127.0.0.1:8000; the unchanged public SQL
uses 0.0.0.0:8000, which must resolve locally without a proxy.

```sh
python3 native_acceptance.py --port 2883 \
  --log /tmp/ai-protocol-requests.jsonl --out /tmp/ai-native-results.json
python3 public_sql.py --source /path/to/seekdb --port 2883 \
  --suite ai_function_enhancement --out /tmp/ai-public-results.json
python3 public_sql.py --source /path/to/seekdb --port 2883 \
  --suite ai_prompt --out /tmp/prompt-public-results.json
```

The public driver reads the repository's original `.test` and `.result` bytes;
it is an explicitly limited PyMySQL driver, not mysqltest. It compares original
SQL errors and result cells (JSON object order/spacing are semantic), preserves
raw BINARY/NUL fields, and stops on unsupported harness instructions. The optional
`--first-connection-only` flag reports the unrun multi-user remainder.
`ai_model_ddl` and `ai_model_lower_case_name_mysql1` cover relevant existing DDL
and name behavior. The fixed public `ai_model_endpoint_ddl` references
`max_concurrency`, which is absent from the fixed source baseline: its initial
segment also fails on the clean baseline, so it must not be reported as passing.
The multi-user portions and mysql0 tenant harness are not covered by this driver.

The acceptance report records request-log byte ranges, SQL/result digests,
actual plans, multi-input request sizes and identity checks. It covers typed and
nested Prompt consumption, out-of-row LONGTEXT, invalid placeholders, config
propagation, response validation, native Ollama and ALTER with empty keys,
47-item rerank, 137 mixed embedding rows, NULL/skip/filter/LIMIT/tails, wrapper
batching, error recovery, and ordinary relational/JSON SQL. Test labels and
synthetic model names occur only in this directory, never in product decisions.
