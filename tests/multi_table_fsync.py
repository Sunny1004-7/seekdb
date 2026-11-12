#!/usr/bin/env python3
"""Does the single-fsync saving survive a multi-table transaction?

Single-table commits fold the marker into the record file (one fsync).
A transaction touching two tables must fall back to the separate commits.log,
so the second file's block cost should come back. Measures bytes per commit.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

BIN = sys.argv[1]
N = 100


def io_bytes(pid):
    r = w = 0
    with open('/proc/%d/io' % pid) as f:
        for ln in f:
            if ln.startswith('read_bytes:'):
                r = int(ln.split()[1])
            elif ln.startswith('write_bytes:'):
                w = int(ln.split()[1])
    return r + w


d = tempfile.mkdtemp(prefix='mvdb-multi-')
try:
    p = subprocess.Popen([BIN, d], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1)
    assert json.loads(p.stdout.readline()) == {'type': 'ready'}
    n = [0]

    def send(sql):
        n[0] += 1
        p.stdin.write(json.dumps({'id': n[0], 'op': 'execute', 'sql': sql, 'search_mode': 'exact'}) + '\n')
        p.stdin.flush()
        r = json.loads(p.stdout.readline())
        assert r.get('ok'), r
        return r

    for t in ('a', 'b'):
        send("CREATE TABLE %s (id BIGINT PRIMARY KEY, v VECTOR(128));" % t)
    vec = '[' + ','.join('0.5' for _ in range(128)) + ']'

    for phase, (label, tables) in enumerate((('single-table', ('a',)), ('two-table', ('a', 'b')))):
        base = io_bytes(p.pid)
        for i in range(N):
            send("BEGIN;")
            for t in tables:
                send("INSERT INTO %s VALUES (%d,'%s');" % (t, phase * 100000 + i + 1 + (0 if t == 'a' else 50000), vec))
            send("COMMIT;")
        used = io_bytes(p.pid) - base
        print('%-14s %7d bytes / %d commits = %6.0f bytes per commit' % (label, used, N, used / N))
    n[0] += 1
    p.stdin.write(json.dumps({'id': n[0], 'op': 'shutdown'}) + '\n')
    p.stdin.flush()
    p.stdout.readline()
    p.wait(timeout=60)
finally:
    shutil.rmtree(d, ignore_errors=True)