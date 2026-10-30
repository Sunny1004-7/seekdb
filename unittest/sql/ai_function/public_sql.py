#!/usr/bin/env python3
"""Run unchanged public SQL with a local PyMySQL driver.

This is not mysqltest or the organizer's service. Compare data with the original
.result, and numeric SQL errors with the original --error directives.
Unsupported harness commands stop parsing rather than silently disappearing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path


def parse_test(path):
    statements = []
    buffer = ""
    quote = None
    escaped = False
    allowed = [0]
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.strip()
        if not buffer.strip():
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("--error "):
                allowed = [int(part.strip()) for part in stripped[8:].split(",")]
                continue
            if stripped in {"--disable_warnings", "--enable_warnings", "--disable_query_log", "--enable_query_log"}:
                continue
            if stripped.startswith("--"):
                raise ValueError(f"Unsupported mysqltest command: {stripped}")
        for char in line:
            buffer += char
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in {"'", '"', "`"}:
                quote = char
            elif char == ";":
                sql = buffer.strip()
                buffer = ""
                if re.fullmatch(r"connection\s+\w+;", sql, flags=re.I):
                    if not re.fullmatch(r"connection\s+default;", sql, flags=re.I):
                        statements.append({"harness": sql})
                    continue
                if re.match(r"connect\s*\(", sql, flags=re.I):
                    statements.append({"harness": sql})
                    continue
                statements.append({"sql": sql, "allowed_errors": allowed})
                allowed = [0]
        if not buffer.strip():
            buffer = ""
    if buffer.strip() or quote:
        raise ValueError("Incomplete SQL at end of " + str(path))
    return statements


def attach_expected(statements, expected_path):
    raw = expected_path.read_text(encoding="utf-8")
    positions = [match.start() for match in re.finditer(r"\S", raw)]
    normalized = "".join(raw[pos] for pos in positions)
    spans = []
    cursor = 0
    sql_statements = [s for s in statements if "sql" in s]
    for statement in sql_statements:
        needle = re.sub(r"\s+", "", statement["sql"])
        found = normalized.find(needle, cursor)
        if found < 0:
            raise ValueError("SQL not found in original .result: " + statement["sql"][:160])
        end = found + len(needle)
        spans.append((positions[found], positions[end - 1] + 1))
        cursor = end
    for index, statement in enumerate(sql_statements):
        start = spans[index][1]
        end = spans[index + 1][0] if index + 1 < len(spans) else len(raw)
        statement["expected_output"] = raw[start:end].strip()


def mysql_cell(value):
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    # These frozen mysqltest .result files contain raw fields, including literal
    # NUL padding in BINARY columns. Escaping again corrupts both JSON escapes
    # (e.g. \u0000) and the ordinary BINARY field beside them.
    return str(value)


def equivalent_cell(actual, expected):
    if actual == expected:
        return True
    # JSON spacing and object key order are not semantic differences.
    try:
        return json.loads(actual) == json.loads(expected)
    except (ValueError, TypeError):
        return False


def compare_rows(rows, columns, expected):
    lines = expected.splitlines()
    if columns is None:
        return expected == ""
    if not lines or len(lines) != len(rows) + 1:
        return False
    expected_header = lines[0].split("\t")
    if [re.sub(r"\s+", "", s) for s in columns] != [re.sub(r"\s+", "", s) for s in expected_header]:
        return False
    for row, line in zip(rows, lines[1:]):
        expected_row = line.split("\t")
        if len(row) != len(expected_row):
            return False
        for cell, reference in zip(row, expected_row):
            if not equivalent_cell(mysql_cell(cell), reference):
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--suite", choices=["ai_function_enhancement", "ai_prompt", "ai_model_ddl", "ai_model_endpoint_ddl", "ai_model_lower_case_name_mysql1"], required=True)
    parser.add_argument("--port", type=int, default=2883)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--first-connection-only", action="store_true",
                        help="Run the initial unchanged SQL segment; explicitly report the unrun multi-user segment")
    args = parser.parse_args()
    suite = args.source / "tools/deploy/mysql_test/test_suite/ai_function"
    test_path = suite / "t" / (args.suite + ".test")
    result_path = suite / "r/mysql" / (args.suite + ".result")
    statements = parse_test(test_path)
    attach_expected(statements, result_path)
    unrun_reason = None
    total_sql_count = sum("sql" in statement for statement in statements)
    harness = next((i for i, statement in enumerate(statements) if "harness" in statement), None)
    if harness is not None:
        if not args.first_connection_only:
            raise ValueError("This suite needs multiple mysqltest sessions; use --first-connection-only for an explicit initial-segment run")
        unrun_reason = "Stopped before harness directive: " + statements[harness]["harness"]
        statements = statements[:harness]
    if args.parse_only:
        print(json.dumps({"suite": args.suite, "statements": len(statements), "all_expected_blocks_located": True,
                          "total_sql": total_sql_count, "unrun_reason": unrun_reason}))
        return
    import pymysql
    connection = pymysql.connect(host="127.0.0.1", port=args.port, user="root", database="test", charset="utf8mb4", autocommit=True, read_timeout=180)
    report = {
        "suite": args.suite,
        "kind": "unchanged official public SQL + local PyMySQL driver + local protocol fixture",
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "cases": [],
        "total_sql": total_sql_count,
        "unrun_reason": unrun_reason,
    }
    with connection, connection.cursor() as cursor:
        cursor.execute("SELECT VERSION()")
        report["server_version"] = cursor.fetchone()[0]
        for index, statement in enumerate(statements):
            started = time.monotonic()
            case = {"index": index, **statement}
            try:
                cursor.execute(statement["sql"])
                columns = [col[0] for col in cursor.description] if cursor.description else None
                rows = cursor.fetchall() if columns is not None else []
                case.update(columns=columns, rows=rows, sql_error=0)
                case["passed"] = 0 in statement["allowed_errors"] and compare_rows(rows, columns, statement["expected_output"])
            except pymysql.Error as exc:
                case.update(sql_error=exc.args[0], error=str(exc))
                case["passed"] = exc.args[0] in statement["allowed_errors"]
            case["elapsed_s"] = time.monotonic() - started
            report["cases"].append(case)
    report["passed"] = sum(case["passed"] for case in report["cases"])
    report["failed"] = len(statements) - report["passed"]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["suite", "server_version", "passed", "failed"]}, ensure_ascii=False))
    if report["failed"]:
        for case in report["cases"]:
            if not case["passed"]:
                print(json.dumps({key: case.get(key) for key in ["index", "sql", "columns", "rows", "error", "expected_output"]}, ensure_ascii=False, default=str))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
