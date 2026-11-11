#!/usr/bin/env python3
"""Split the per-query cost into parse/protocol vs scan, using LIMIT 0.

LIMIT 0 goes through the same JSON parse, SQL lexer/parser and query-vector
literal parse, but the scan loop is skipped entirely, so it measures the floor.
"""
import json
import subprocess
import sys
import time

BIN, SQL, TRUTH = sys.argv[1], sys.argv[2], sys.argv[3]
truth = json.load(open(TRUTH, encoding='utf-8'))
stmts = [l.strip() for l in open(SQL, encoding='utf-8') if l.strip()]

import tempfile, shutil, os
d = tempfile.mkdtemp(prefix='mvdb-split-')


class P:
    def __init__(self):
        self.p = subprocess.Popen([BIN, d], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1)
        assert json.loads(self.p.stdout.readline()) == {'type': 'ready'}
        self.n = 0

    def send(self, sql):
        self.n += 1
        self.p.stdin.write(json.dumps({'id': self.n, 'op': 'execute', 'sql': sql,
                                       'search_mode': 'exact'}) + '\n')
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())


try:
    p = P()
    for s in stmts:
        assert p.send(s).get('ok')
    qs = [t['sql'] for t in truth]
    for label, suffix in (('LIMIT 0  (parse+protocol only)', 'LIMIT 0;'),
                          ('LIMIT 10 (parse+protocol+scan)', 'LIMIT 10;')):
        mod = [q.replace('LIMIT 10;', suffix) for q in qs]
        for q in mod[:50]:
            p.send(q)
        t0 = time.time()
        for q in mod:
            r = p.send(q)
            assert r.get('ok'), r
        dt = time.time() - t0
        print('%-34s %7.3f ms/query   (%d queries, %.3f s)' % (label, dt * 1000 / len(mod), len(mod), dt))
    # also a short SQL with no vector literal, to separate literal parsing
    short = ['SELECT id FROM big LIMIT 1;'] * 1000
    for q in short[:50]:
        p.send(q)
    t0 = time.time()
    for q in short:
        p.send(q)
    dt = time.time() - t0
    print('%-34s %7.3f ms/query' % ('short SQL, no vector literal', dt * 1000 / len(short)))
finally:
    shutil.rmtree(d, ignore_errors=True)