#!/usr/bin/env python3
"""Official-shaped benchmark: 10k rows x 128 dims, 100-row write batches,
1000 TopK queries (K=10, half with an equality filter).

Reports T (wall clock), B (process read/write bytes), M (peak RSS) and verifies
every answer against brute-force truth. The official metric also counts page
cache in M and uses cgroup io.stat for B, so these local numbers are a lower
bound on M and an approximation of B.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

BIN = sys.argv[1]
SQL = sys.argv[2]
TRUTH = sys.argv[3]


def io_bytes(pid):
    r = w = 0
    try:
        with open('/proc/%d/io' % pid) as f:
            for ln in f:
                if ln.startswith('read_bytes:'):
                    r = int(ln.split()[1])
                elif ln.startswith('write_bytes:'):
                    w = int(ln.split()[1])
    except OSError:
        pass
    return r, w


def peak_mb(pid):
    try:
        with open('/proc/%d/status' % pid) as f:
            for ln in f:
                if ln.startswith('VmHWM:'):
                    return int(ln.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


class Proc:
    def __init__(self, data):
        self.p = subprocess.Popen([BIN, data], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1)
        assert json.loads(self.p.stdout.readline()) == {'type': 'ready'}
        self.n = 0

    def send(self, sql, mode='exact'):
        self.n += 1
        self.p.stdin.write(json.dumps({'id': self.n, 'op': 'execute', 'sql': sql,
                                       'search_mode': mode}) + '\n')
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        self.n += 1
        self.p.stdin.write(json.dumps({'id': self.n, 'op': 'shutdown'}) + '\n')
        self.p.stdin.flush()
        self.p.stdout.readline()
        self.p.wait(timeout=120)


truth = json.load(open(TRUTH, encoding='utf-8'))
stmts = [l.strip() for l in open(SQL, encoding='utf-8') if l.strip()]
d = tempfile.mkdtemp(prefix='mvdb-bench-')
try:
    p = Proc(d)
    r0, w0 = io_bytes(p.p.pid)
    t0 = time.time()
    for s in stmts:
        r = p.send(s)
        assert r.get('ok'), r
    t_write = time.time() - t0
    r1, w1 = io_bytes(p.p.pid)
    b_write = (r1 - r0) + (w1 - w0)
    m_write = peak_mb(p.p.pid)

    t0 = time.time()
    bad = 0
    for t in truth:
        r = p.send(t['sql'])
        assert r.get('ok'), r
        if [x[0] for x in r['rows']] != t['ids']:
            bad += 1
    t_query = time.time() - t0
    r2, w2 = io_bytes(p.p.pid)
    b_query = (r2 - r1) + (w2 - w1)
    m_total = peak_mb(p.p.pid)
    disk = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    p.close()

    print('write : T=%.3fs  B=%.2f MiB  M=%.1f MiB  (%d statements)'
          % (t_write, b_write / 1048576, m_write, len(stmts)))
    print('query : T=%.3fs  B=%.2f MiB  M=%.1f MiB  (%d queries, %.3f ms/query)'
          % (t_query, b_query / 1048576, m_total, len(truth), t_query * 1000 / len(truth)))
    print('disk  : %.2f MiB   correctness: %d/%d exact'
          % (disk / 1048576, len(truth) - bad, len(truth)))
    print('recall@10 = %.4f' % ((len(truth) - bad) / len(truth)))
    out = {'t_write': t_write, 'b_write': b_write, 'm_write_mib': m_write,
           't_query': t_query, 'b_query': b_query, 'm_peak_mib': m_total,
           'disk_mib': disk / 1048576, 'exact': len(truth) - bad, 'nq': len(truth)}
    print('JSON ' + json.dumps(out))
    sys.exit(1 if bad else 0)
finally:
    shutil.rmtree(d, ignore_errors=True)