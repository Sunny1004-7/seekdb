#!/usr/bin/env python3
"""Big-table test: on-disk data far exceeds the process memory budget.

Usage: big_table_test.py <bin> <sql_file> <truth_json> <cap_mb>
The SQL/truth files come from build/gen_truth. Checks exact + filtered TopK
against brute-force truth, then recovery after a restart. Reports peak RSS.
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

BIN, SQL, TRUTH = sys.argv[1], sys.argv[2], sys.argv[3]
CAP_MB = int(sys.argv[4]) if len(sys.argv) > 4 else 64


def limit():
    resource.setrlimit(resource.RLIMIT_AS, (CAP_MB * 1024 * 1024, CAP_MB * 1024 * 1024))


class Proc:
    def __init__(self, data):
        self.p = subprocess.Popen([BIN, data], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                  bufsize=1, preexec_fn=limit)
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('no ready: ' + self.p.stderr.read()[:500])
        assert json.loads(line) == {'type': 'ready'}, line
        self.n = 0

    def send(self, sql, mode='exact'):
        self.n += 1
        self.p.stdin.write(json.dumps({'id': self.n, 'op': 'execute', 'sql': sql,
                                       'search_mode': mode}) + '\n')
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('died: ' + self.p.stderr.read()[:500])
        return json.loads(line)

    def peak_mb(self):
        try:
            with open('/proc/%d/status' % self.p.pid) as f:
                for ln in f:
                    if ln.startswith('VmHWM:'):
                        return int(ln.split()[1]) / 1024.0
        except OSError:
            pass
        return -1.0

    def close(self):
        self.n += 1
        self.p.stdin.write(json.dumps({'id': self.n, 'op': 'shutdown'}) + '\n')
        self.p.stdin.flush()
        self.p.stdout.readline()
        self.p.wait(timeout=180)


truth = json.load(open(TRUTH, encoding='utf-8'))
d = tempfile.mkdtemp(prefix='mvdb-big-')
fails = []
try:
    p = Proc(d)
    t0 = time.time()
    nstmt = 0
    with open(SQL, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = p.send(line)
            if not r.get('ok'):
                raise RuntimeError('stmt %d failed: %s' % (nstmt, r))
            nstmt += 1
    write_s = time.time() - t0
    peak_w = p.peak_mb()
    disk = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    print('load: %d statements in %.1fs | on-disk %.1f MiB | peak RSS %.1f MiB | cap %d MiB'
          % (nstmt, write_s, disk / 1048576, peak_w, CAP_MB))
    if disk / 1048576 <= CAP_MB:
        fails.append('data did not exceed the memory cap')
    if peak_w > CAP_MB:
        fails.append('peak RSS exceeded cap during load')

    t0 = time.time()
    bad = 0
    for ti, t in enumerate(truth):
        r = p.send(t['sql'])
        if not r.get('ok'):
            raise RuntimeError('query failed: %s' % r)
        got = [row[0] for row in r['rows']]
        if got != t['ids']:
            bad += 1
            if bad <= 2:
                print('  mismatch #%d got=%s want=%s' % (ti, got[:5], t['ids'][:5]))
    q_s = time.time() - t0
    print('query: %d TopK in %.2fs (%.1f ms/query) | peak RSS %.1f MiB'
          % (len(truth), q_s, q_s * 1000 / len(truth), p.peak_mb()))
    if bad:
        fails.append('%d/%d TopK mismatches' % (bad, len(truth)))
    else:
        print('PASS exact + filtered TopK match brute-force truth (%d/%d)' % (len(truth), len(truth)))
    p.close()

    t0 = time.time()
    p2 = Proc(d)
    rec_s = time.time() - t0
    r = p2.send(truth[0]['sql'])
    ok = r.get('ok') and [x[0] for x in r['rows']] == truth[0]['ids']
    print(('PASS' if ok else 'FAIL') + ' recovery + TopK after restart (ready in %.2fs, peak RSS %.1f MiB)'
          % (rec_s, p2.peak_mb()))
    if not ok:
        fails.append('recovery topk')
    p2.close()
finally:
    shutil.rmtree(d, ignore_errors=True)

print('---')
print('big-table suite: %d failure(s)' % len(fails))
sys.exit(1 if fails else 0)