#!/usr/bin/env python3
"""Fault injection the public cases do not cover: SIGKILL mid-transaction,
SIGKILL during recovery, and durability of an acknowledged COMMIT."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

BIN = sys.argv[1] if len(sys.argv) > 1 else './build/mvdb'


class Proc:
    def __init__(self, data):
        self.p = subprocess.Popen([BIN, data], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding='utf-8', bufsize=1)
        line = self.p.stdout.readline()
        assert json.loads(line) == {'type': 'ready'}, line
        self.n = 0

    def send(self, sql, mode='exact'):
        self.n += 1
        self.p.stdin.write(json.dumps({'id': self.n, 'op': 'execute', 'sql': sql,
                                       'search_mode': mode}) + '\n')
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def kill(self):
        self.p.send_signal(signal.SIGKILL)
        self.p.wait()

    def close(self):
        self.n += 1
        self.p.stdin.write(json.dumps({'id': self.n, 'op': 'shutdown'}) + '\n')
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        assert r['ok'] is True, r
        assert self.p.wait(timeout=30) == 0


def rows(r):
    assert r['ok'] is True, r
    return r['rows']


fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else '  ' + str(detail)))
    if not cond:
        fails.append(name)


d = tempfile.mkdtemp(prefix='mvdb-crash-')
try:
    # 1. acknowledged COMMIT survives SIGKILL; uncommitted txn does not
    p = Proc(d)
    p.send("CREATE TABLE t (id BIGINT PRIMARY KEY, c VARCHAR(8), v VECTOR(2));")
    p.send("INSERT INTO t VALUES (1,'a','[1,0]'),(2,'b','[0,1]');")
    p.send("BEGIN;")
    p.send("INSERT INTO t VALUES (3,'c','[1,1]');")
    p.send("COMMIT;")
    p.send("BEGIN;")
    p.send("INSERT INTO t VALUES (4,'d','[2,2]');")
    p.send("DELETE FROM t WHERE id = 1;")
    p.kill()

    p = Proc(d)
    got = rows(p.send("SELECT id FROM t ORDER BY id;"))
    check('committed survives kill, uncommitted discarded', got == [[1], [2], [3]], got)
    got = rows(p.send("SELECT id, L2_DISTANCE(v,'[0,0]') AS d FROM t ORDER BY d LIMIT 5;"))
    check('vector search consistent after recovery', [r[0] for r in got] == [1, 2, 3], got)

    # 2. kill during recovery, then recover again (repeated recovery)
    p.send("BEGIN;")
    p.send("INSERT INTO t VALUES (5,'e','[3,3]');")
    p.send("COMMIT;")
    p.kill()
    p2 = Proc(d)
    p2.kill()          # killed right after ready, mid-life
    p3 = Proc(d)
    got = rows(p3.send("SELECT id FROM t ORDER BY id;"))
    check('repeated recovery is idempotent', got == [[1], [2], [3], [5]], got)

    # 3. statement failure inside a txn, then COMMIT (compensation must hold)
    p3.send("BEGIN;")
    p3.send("UPDATE t SET c='z' WHERE id=2;")
    r = p3.send("INSERT INTO t VALUES (9,'x','[0,0]'),(3,'dup','[0,0]');")
    check('duplicate reported', r.get('ok') is False and r['error']['code'] == 'duplicate_key', r)
    p3.send("COMMIT;")
    p3.kill()
    p4 = Proc(d)
    got = rows(p4.send("SELECT id, c FROM t ORDER BY id;"))
    check('aborted statement stays aborted after commit+kill',
          got == [[1, 'a'], [2, 'z'], [3, 'c'], [5, 'e']], got)

    # 4. rollback then kill: nothing from the txn may appear
    p4.send("BEGIN;")
    p4.send("DELETE FROM t WHERE id >= 1;")
    p4.send("ROLLBACK;")
    p4.kill()
    p5 = Proc(d)
    got = rows(p5.send("SELECT id FROM t ORDER BY id;"))
    check('rollback then kill keeps rows', got == [[1], [2], [3], [5]], got)
    p5.close()
finally:
    shutil.rmtree(d, ignore_errors=True)

print('---')
print('crash suite: %d failure(s)' % len(fails))
sys.exit(1 if fails else 0)