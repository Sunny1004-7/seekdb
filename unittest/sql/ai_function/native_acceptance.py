#!/usr/bin/env python3
"""Native SQL/request assertions against an independent local protocol fixture.

This is supplemental coverage, not official hidden tests. PyMySQL is a test
client only. All model names and faults below are synthetic test inputs.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def vector(text, dimensions):
    digest = hashlib.sha256(text.encode()).digest()
    return [int.from_bytes(digest[2 * (i % 16):2 * (i % 16) + 2], 'big') / 8.0
            for i in range(dimensions)]


def decode(value):
    return json.loads(value.decode() if isinstance(value, bytes) else value)


def completion(value):
    try:
        return decode(value)
    except ValueError:
        return value


class Run:
    def __init__(self, connection, log, report):
        self.connection, self.log, self.report = connection, log, report
        self.current = None

    def sql(self, statement, params=None):
        started = time.monotonic()
        with self.connection.cursor() as cursor:
            try:
                cursor.execute(statement, params)
                rows = cursor.fetchall() if cursor.description else []
                record = {'sql': statement, 'parameters_sha256': hashlib.sha256(
                    json.dumps(params, ensure_ascii=False, default=str).encode()).hexdigest(),
                    'rows': len(rows), 'rows_sha256': hashlib.sha256(
                    json.dumps(rows, ensure_ascii=False, default=str).encode()).hexdigest(),
                    'elapsed_s': time.monotonic() - started, 'error': 0}
                if self.current is not None:
                    self.current['queries'].append(record)
                return rows
            except Exception as exc:
                if self.current is not None:
                    self.current['queries'].append({'sql': statement, 'error': str(exc),
                                                   'elapsed_s': time.monotonic() - started})
                raise

    def value(self, statement, params=None):
        rows = self.sql(statement, params)
        assert len(rows) == 1 and len(rows[0]) == 1, rows
        return rows[0][0]

    def position(self):
        return self.log.stat().st_size if self.log.exists() else 0

    def calls(self, start=None):
        with self.log.open('rb') as stream:
            stream.seek(self.current['request_log_start'] if start is None else start)
            return [json.loads(line) for line in stream if line.strip()]

    def case(self, name, function):
        record = {'name': name, 'queries': [], 'request_log_start': self.position()}
        self.current = record
        started = time.monotonic()
        try:
            function()
            record['passed'] = True
        except Exception as exc:
            record.update(passed=False, error=repr(exc))
        record.update(elapsed_s=time.monotonic() - started, request_log_end=self.position())
        self.report['cases'].append(record)
        self.current = None
        print(json.dumps({k: record[k] for k in ['name', 'passed', 'elapsed_s']}
                         | ({'error': record['error']} if 'error' in record else {})), flush=True)

    def expect_sql_error(self, statement, params=None):
        import pymysql
        try:
            self.sql(statement, params)
        except pymysql.Error:
            return
        raise AssertionError('Expected a SQL error: ' + statement)

    def model(self, name, kind, provider, path, key='local-test-only'):
        self.sql('CALL DBMS_AI_SERVICE.CREATE_AI_MODEL(%s,%s)',
                 (name, json.dumps({'type': kind, 'model_name': name})))
        self.sql('CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT(%s,%s)',
                 (name + '_ep', json.dumps({'ai_model_name': name, 'scope': 'all',
                   'url': 'http://127.0.0.1:8000' + path, 'provider': provider, 'access_key': key})))

    def endpoint(self, name, path):
        self.sql('CALL DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT(%s,%s)',
                 (name + '_ep', json.dumps({'url': 'http://127.0.0.1:8000' + path})))


def foundations(run):
    def setup():
        for name, kind, path in [('r10_complete', 'completion', '/v1/chat/completions'),
                                 ('r10_embed', 'dense_embedding', '/v1/embeddings')]:
            for sql, arg in [('DROP_AI_MODEL_ENDPOINT', name + '_ep'), ('DROP_AI_MODEL', name)]:
                try:
                    run.sql('CALL DBMS_AI_SERVICE.' + sql + '(%s)', (arg,))
                except Exception:
                    pass
            run.model(name, kind, 'openai', path)
    run.case('foundations/setup', setup)

    def prompt_json():
        result = decode(run.value("SELECT AI_PROMPT('{0}/{0}/{1}', JSON_ARRAY(1,true), JSON_OBJECT('k',2))"))
        assert result == {'args': [[1, True], {'k': 2}], 'template': '{0}/{0}/{1}'}, result
        # The official constructor permits unbound placeholders; expansion checks them.
        assert decode(run.value("SELECT AI_PROMPT('{0}/{1}', 'one')"))['args'] == ['one']
    run.case('K01-1/typed-json-construction', prompt_json)

    def render():
        text = completion(run.value("""SELECT AI_COMPLETE('r10_complete',
             AI_PROMPT('{0}|{0}|{1}|{2}', JSON_ARRAY(1,true), JSON_OBJECT('k',2),
                        AI_PROMPT('inner:{0}', '你好\\n\\"世界')), NULL)"""))
        assert text.startswith('echo:')
        parts = text[5:].split('|')
        assert len(parts) == 4 and decode(parts[0]) == [1, True] and parts[0] == parts[1], text
        assert decode(parts[2]) == {'k': 2} and parts[3] == 'inner:你好\n"世界', text
        received = run.calls()[-1]['body']['messages'][-1]['content']
        assert received == text[5:], received
    run.case('K01-1/nested-repeated-unicode-consumption', render)

    def lob():
        run.sql('DROP TABLE IF EXISTS r10_lob')
        run.sql('CREATE TABLE r10_lob(id INT PRIMARY KEY,t LONGTEXT,a LONGTEXT)')
        template = '前缀' * 3000 + '{0}/' + '{0}' * 9 + '尾部'
        argument = '世界\\"\n' * 2500
        run.sql('INSERT INTO r10_lob VALUES(1,%s,%s)', (template, argument))
        prompt = decode(run.value('SELECT AI_PROMPT(t,a) FROM r10_lob'))
        assert prompt == {'args': [argument], 'template': template}
        expected = template.replace('{0}', argument)
        output = completion(run.value("SELECT AI_COMPLETE('r10_complete',AI_PROMPT(t,a)) FROM r10_lob"))
        assert output == 'echo:' + expected, (len(output), len(expected))
        assert run.calls()[-1]['body']['messages'][-1]['content'] == expected
        run.sql('DROP TABLE r10_lob')
    run.case('K01-1/out-of-row-longtext-full-consumption', lob)

    def invalid_placeholders():
        for template in ['{-1}', '{999}', '{99999999999999999999999999999999999999}']:
            before = run.position()
            run.expect_sql_error("SELECT AI_COMPLETE('r10_complete',AI_PROMPT(%s,'x'))", (template,))
            assert run.position() == before
        assert completion(run.value("SELECT AI_COMPLETE('r10_complete',AI_PROMPT('literal {text} {-1 days} {0}','x'))")) == 'echo:literal {text} {-1 days} x'
    run.case('K01-1/invalid-indices-before-network', invalid_placeholders)

    def configurations():
        for config in [None, '{"temperature":0.125,"max_tokens":19}']:
            assert completion(run.value("SELECT AI_COMPLETE('r10_complete','config',%s)", (config,))) == 'echo:config'
        assert completion(run.value("SELECT AI_COMPLETE('r10_complete','object',JSON_OBJECT('temperature',0.25,'max_tokens',23))")) == 'echo:object'
        calls = run.calls()
        assert 'temperature' not in calls[0]['body']
        assert calls[1]['body']['temperature'] == 0.125 and calls[1]['body']['max_tokens'] == 19
        assert calls[2]['body']['temperature'] == 0.25 and calls[2]['body']['max_tokens'] == 23
        run.expect_sql_error("SELECT AI_COMPLETE('r10_complete','config',JSON_ARRAY(1,2))")
    run.case('K01-3/null-string-and-object-config', configurations)

    def embedding():
        for text, dimension in [('默认维度', None), ('explicit-three', 3), ('explicit-eleven', 11)]:
            sql = "SELECT AI_EMBED('r10_embed',%s)" if dimension is None else "SELECT AI_EMBED('r10_embed',%s,%s)"
            args = (text,) if dimension is None else (text, dimension)
            output = decode(run.value(sql, args))
            assert output == vector(text, 7 if dimension is None else dimension), output
        calls = run.calls()
        assert 'dimensions' not in calls[0]['body'] and calls[1]['body']['dimensions'] == 3
        for dimension in [0, -1, 0.5, 'three', None]:
            run.expect_sql_error("SELECT AI_EMBED('r10_embed','invalid',%s)", (dimension,))
        before = run.position()
        assert run.value("SELECT AI_EMBED('r10_embed',NULL)") is None
        run.expect_sql_error("SELECT AI_EMBED(NULL,'content')")
        assert run.position() == before
        run.expect_sql_error("SELECT AI_EMBED('r10_embed','')")
    run.case('K01-2/dimensions-vector-and-null', embedding)


def protocols(run):
    def setup():
        for name, kind, provider, path, key in [
            ('r10_ollama_c', 'completion', 'oLlAmA', '/api/generate', ''),
            ('r10_ollama_e', 'dense_embedding', 'OLLAMA', '/api/embed', ''),
            ('r10_rerank', 'rerank', 'siliconflow', '/v1/rerank', 'local-test-only'),
        ]:
            for sql, arg in [('DROP_AI_MODEL_ENDPOINT', name + '_ep'), ('DROP_AI_MODEL', name)]:
                try:
                    run.sql('CALL DBMS_AI_SERVICE.' + sql + '(%s)', (arg,))
                except Exception:
                    pass
            run.model(name, kind, provider, path, key)
    run.case('protocols/setup', setup)

    def ollama():
        assert run.value("SELECT provider FROM oceanbase.DBA_OB_AI_MODEL_ENDPOINTS WHERE endpoint_name='r10_ollama_c_ep'") == 'oLlAmA'
        run.sql("CALL DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT('r10_ollama_c_ep','{\"provider\":\"Ollama\"}')")
        assert run.value("SELECT provider FROM oceanbase.DBA_OB_AI_MODEL_ENDPOINTS WHERE endpoint_name='r10_ollama_c_ep'") == 'Ollama'
        assert completion(run.value("SELECT AI_COMPLETE('r10_ollama_c','native-generation')")) == 'echo:native-generation'
        assert decode(run.value("SELECT AI_EMBED('r10_ollama_e','native-embedding',9)")) == vector('native-embedding', 9)
        calls = run.calls()
        assert [c['path'] for c in calls] == ['/api/generate', '/api/embed']
        assert calls[0]['body']['stream'] is False
        assert not any(c['has_authorization'] for c in calls)
        assert all(c['content_type'].startswith('application/json') for c in calls)
        # Empty credentials are specific to Ollama; other providers retain their contract.
        run.expect_sql_error("CALL DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT('r10_ollama_c_ep','{\"provider\":\"openai\"}')")
    run.case('K01-4/native-protocol-case-and-empty-key', ollama)

    def rerank():
        scores = [float(((i * 17) % 31) - 15) for i in range(47)]
        scores[0] = scores[21] = scores[46] = 100.0
        documents = [f'doc-{i}:score:{score}' for i, score in enumerate(scores)]
        expected = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
        result = decode(run.value("SELECT AI_RERANK('r10_rerank','query',%s,NULL)", (json.dumps(documents),)))
        assert [r['index'] for r in result] == expected
        assert [r['relevance_score'] for r in result] == [scores[i] for i in expected]
        calls = run.calls()
        assert [len(c['body']['documents']) for c in calls] == [20, 20, 7]
        assert [t for c in calls for t in c['body']['documents']] == documents
        objects = [{'id': f'id-{i}', 'body': documents[i], 'extra': {'source': i}}
                   for i in range(len(documents))]
        # Identical text must not merge distinct source IDs.
        objects[22]['body'] = objects[0]['body']
        scores[22] = scores[0]
        expected = sorted(range(len(objects)), key=lambda i: (-scores[i], i))
        start = run.position()
        result = decode(run.value("SELECT AI_RERANK('r10_rerank','query',%s,'body')", (json.dumps(objects),)))
        assert [r['id'] for r in result] == [objects[i]['id'] for i in expected]
        for actual, index in zip(result, expected):
            assert actual.pop('_model_score') == scores[index]
            assert actual == objects[index], (actual, objects[index])
        assert [len(c['body']['documents']) for c in run.calls(start)] == [20, 20, 7]
    run.case('K01-5/47-items-global-stable-identity', rerank)

    def faults():
        paths = [
            ('r10_complete', '/v1/chat/completions', ['missing', 'empty', 'nonstring', 'root_array'],
             "SELECT AI_COMPLETE('r10_complete','fault')"),
            ('r10_embed', '/v1/embeddings', ['missing', 'count', 'duplicate', 'index', 'index_type',
                                           'empty', 'element', 'boolean', 'dimension', 'root_array'],
             "SELECT AI_EMBED('r10_embed','fault',5)"),
            ('r10_rerank', '/v1/rerank', ['missing', 'count', 'duplicate', 'index', 'score', 'root_array'],
             "SELECT AI_RERANK('r10_rerank','q','[\"a\",\"b\",\"c\"]')"),
        ]
        for model, normal, names, sql in paths:
            try:
                for name in names:
                    run.endpoint(model, '/fault/' + name + normal)
                    run.expect_sql_error(sql)
                    assert run.value('SELECT 1+2') == 3
            finally:
                run.endpoint(model, normal)
        assert completion(run.value("SELECT AI_COMPLETE('r10_complete','recovered')")) == 'echo:recovered'
        assert decode(run.value("SELECT AI_EMBED('r10_embed','recovered',5)")) == vector('recovered', 5)
    run.case('failure-paths/invalid-responses-error-and-recovery', faults)

    def network_failure():
        try:
            run.endpoint('r10_complete', '/fault/http403/v1/chat/completions')
            run.expect_sql_error("SELECT AI_COMPLETE('r10_complete','forbidden')")
            assert run.calls()[-1]['status'] == 403
            run.sql('CALL DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT(%s,%s)',
                    ('r10_complete_ep', json.dumps({'url': 'http://127.0.0.1:9/v1/chat/completions'})))
            run.expect_sql_error("SELECT AI_COMPLETE('r10_complete','unreachable')")
        finally:
            run.endpoint('r10_complete', '/v1/chat/completions')
        assert completion(run.value("SELECT AI_COMPLETE('r10_complete','network-recovered')")) == 'echo:network-recovered'
    run.case('failure-paths/http403-and-connection-refused', network_failure)


def batch(run):
    rows = [(i, None if i % 11 == 0 else 'row-identity-' + str(i),
             'r10_embed_b' if i % 3 == 0 else 'r10_embed', 3 if i % 2 == 0 else 9)
            for i in range(137)]
    by_text = {r[1]: (r[2], r[3]) for r in rows if r[1] is not None}
    by_id = {r[0]: r for r in rows}

    def setup():
        for sql, arg in [('DROP_AI_MODEL_ENDPOINT', 'r10_embed_b_ep'), ('DROP_AI_MODEL', 'r10_embed_b')]:
            try:
                run.sql('CALL DBMS_AI_SERVICE.' + sql + '(%s)', (arg,))
            except Exception:
                pass
        run.model('r10_embed_b', 'dense_embedding', 'openai', '/v1/embeddings')
        run.sql('DROP TABLE IF EXISTS r10_batch')
        run.sql('ALTER SYSTEM SET _rowsets_max_rows=16')
        run.report['batch_max_rows_requested'] = 16
        run.sql('CREATE TABLE r10_batch(id INT PRIMARY KEY,content TEXT,model VARCHAR(32),dim INT)')
        for offset in range(0, len(rows), 40):
            part = rows[offset:offset + 40]
            run.sql('INSERT INTO r10_batch VALUES ' + ','.join(['(%s,%s,%s,%s)'] * len(part)),
                    tuple(v for row in part for v in row))
    run.case('batch/setup', setup)

    def assert_rows(actual, expected_ids):
        assert [r[0] for r in actual] == expected_ids
        for identity, raw in actual:
            _, text, _, dimensions = by_id[identity]
            assert raw is None if text is None else decode(raw) == vector(text, dimensions), identity

    def full_scan():
        query = 'SELECT id,AI_EMBED(model,content,dim) FROM r10_batch ORDER BY id'
        run.report.setdefault('plans', {})['mixed_batch'] = run.sql('EXPLAIN EXTENDED ' + query)
        actual = run.sql(query)
        assert_rows(actual, list(range(137)))
        calls = run.calls()
        assert any(len(c['body']['input']) > 1 for c in calls), 'No actual multi-input request'
        assert len(calls) > 1, 'Expected multiple compatible batches'
        observed = []
        for call in calls:
            assert call['status'] == 200
            body = call['body']
            assert call['response']['data'][0]['index'] == len(body['input']) - 1
            for text in body['input']:
                assert (body['model'], body['dimensions']) == by_text[text]
                observed.append(text)
        assert sorted(observed) == sorted(by_text), 'Missing, duplicated or NULL request rows'
        run.current['request_sizes'] = [len(c['body']['input']) for c in calls]
        run.current['identity_rows_verified'] = len(actual)
    run.case('K01-6/multi-input-reverse-index-mixed-groups-and-tail', full_scan)

    def wrapped_scan():
        actual = run.sql('SELECT id,JSON_LENGTH(AI_EMBED(model,content,dim)) FROM r10_batch ORDER BY id')
        assert actual == tuple((r[0], None if r[1] is None else r[3]) for r in rows), actual[:5]
        calls = run.calls()
        run.current['request_sizes'] = [len(c['body']['input']) for c in calls]
        assert any(len(c['body']['input']) > 1 for c in calls), 'JSON_LENGTH wrapper bypasses multi-input evaluation'
    run.case('K01-6/public-json-length-wrapper-batching', wrapped_scan)

    def null_and_skip():
        before = run.position()
        actual = run.sql('SELECT id,AI_EMBED(model,content,dim) FROM r10_batch WHERE content IS NULL ORDER BY id')
        assert_rows(actual, [r[0] for r in rows if r[1] is None])
        assert run.position() == before, 'All-NULL batches must not reach the service'
        actual = run.sql('SELECT id,AI_EMBED(model,content,dim) FROM r10_batch WHERE id < 0')
        assert actual == () or actual == []
        assert run.position() == before
    run.case('K01-6/all-null-and-empty-scan', null_and_skip)

    def filtered_limit():
        query = 'SELECT id,AI_EMBED(model,content,dim) FROM r10_batch WHERE MOD(id,3)=1 ORDER BY id LIMIT 11'
        run.report.setdefault('plans', {})['filter_limit'] = run.sql('EXPLAIN EXTENDED ' + query)
        expected = [r[0] for r in rows if r[0] % 3 == 1][:11]
        actual = run.sql(query)
        assert_rows(actual, expected)
        calls = run.calls()
        observed = [text for c in calls for text in c['body']['input']]
        for text in observed:
            identity = int(text.rsplit('-', 1)[1])
            assert identity % 3 == 1, ('filter leaked', identity)
        # LIMIT can stop after executor batch evaluation; record actual scope.
        required = {by_id[i][1] for i in expected if by_id[i][1] is not None}
        assert required.issubset(set(observed))
        run.current['returned_ids'] = expected
        run.current['requested_ids'] = [int(t.rsplit('-', 1)[1]) for t in observed]
    run.case('K01-6/filter-and-limit-with-observed-scope', filtered_limit)

    def scalar_equivalence():
        for identity in [1, 3, 20, 57, 135, 136]:
            _, text, model, dimension = by_id[identity]
            scalar = run.value('SELECT AI_EMBED(%s,%s,%s)', (model, text, dimension))
            column = run.value('SELECT AI_EMBED(model,content,dim) FROM r10_batch WHERE id=%s', (identity,))
            assert decode(scalar) == decode(column) == vector(text, dimension)
    run.case('K01-6/scalar-batch-equivalence', scalar_equivalence)

    def conditional_skip():
        actual = run.sql('SELECT id,CASE WHEN MOD(id,5)=0 THEN AI_EMBED(model,content,dim) ELSE NULL END FROM r10_batch ORDER BY id')
        for identity, raw in actual:
            _, text, _, dimensions = by_id[identity]
            if identity % 5 != 0 or text is None:
                assert raw is None
            else:
                assert decode(raw) == vector(text, dimensions)
        observed = [t for c in run.calls() for t in c['body']['input']]
        assert sorted(observed) == sorted(r[1] for r in rows if r[0] % 5 == 0 and r[1] is not None)
        start = run.position()
        wrapped = run.sql('SELECT id,CASE WHEN MOD(id,5)=0 THEN JSON_LENGTH(AI_EMBED(model,content,dim)) ELSE NULL END FROM r10_batch ORDER BY id')
        assert wrapped == tuple((r[0], r[3] if r[0] % 5 == 0 and r[1] is not None else None) for r in rows)
        observed = [t for c in run.calls(start) for t in c['body']['input']]
        assert sorted(observed) == sorted(r[1] for r in rows if r[0] % 5 == 0 and r[1] is not None)
    run.case('K01-6/case-expression-private-skip', conditional_skip)

    def batch_fault():
        run.endpoint('r10_embed', '/fault/mixed_dimension/v1/embeddings')
        try:
            run.expect_sql_error("SELECT id,AI_EMBED('r10_embed',content,7) FROM r10_batch WHERE content IS NOT NULL")
            assert any(len(c['body']['input']) > 1 for c in run.calls())
        finally:
            run.endpoint('r10_embed', '/v1/embeddings')
        actual = run.sql('SELECT id,AI_EMBED(model,content,dim) FROM r10_batch WHERE id BETWEEN 120 AND 136 ORDER BY id')
        assert_rows(actual, list(range(120, 137)))
    run.case('K01-6/batch-response-error-and-recovery', batch_fault)
    run.case('batch/restore-default-batch-size', lambda: run.sql('ALTER SYSTEM SET _rowsets_max_rows=256'))


def ordinary(run):
    def scalar_json():
        assert run.sql("SELECT 2+3,CONCAT('你','好'),COALESCE(NULL,7),NULL IS NULL") == ((5, '你好', 7, 1),)
        assert decode(run.value("SELECT JSON_OBJECT('a',JSON_ARRAY(1,NULL,'你好'),'n',-2.5)")) == {'a': [1, None, '你好'], 'n': -2.5}
        assert run.value("SELECT JSON_LENGTH('{\"a\":[1,2,3],\"b\":null}')") == 2
        assert run.value("SELECT JSON_LENGTH('{\"a\":[1,2,3]}','$.a')") == 3
        assert run.value('SELECT JSON_LENGTH(NULL)') is None
        assert run.value("SELECT JSON_VALID('not-json')") == 0
        run.expect_sql_error("SELECT JSON_LENGTH('not-json')")
    run.case('ordinary/sql-null-unicode-and-json', scalar_json)

    def json_length_batches():
        run.sql('DROP TABLE IF EXISTS r10_json_length')
        run.sql('CREATE TABLE r10_json_length(id INT PRIMARY KEY,doc LONGTEXT,path VARCHAR(32))')
        run.sql('INSERT INTO r10_json_length VALUES(1,%s,%s),(2,NULL,%s),(3,%s,%s),(4,%s,%s)',
                ('{"a":[1,2,3]}', '$.a', '$.a', '{"b":{"k":1}}', '$.b', 'invalid-json', '$.a'))
        assert run.sql('SELECT id,JSON_LENGTH(doc,path) FROM r10_json_length WHERE id<4 ORDER BY id') == ((1,3),(2,None),(3,1))
        assert run.sql('SELECT id,CASE WHEN id<4 THEN JSON_LENGTH(doc,path) ELSE 99 END FROM r10_json_length ORDER BY id') == ((1,3),(2,None),(3,1),(4,99))
        run.expect_sql_error('SELECT JSON_LENGTH(doc,path) FROM r10_json_length WHERE id=4')
        run.sql('DROP TABLE r10_json_length')
    run.case('ordinary/json-length-batches-paths-null-and-private-skip', json_length_batches)

    def relational_lob():
        run.sql('DROP TABLE IF EXISTS r10_ordinary')
        run.sql('CREATE TABLE r10_ordinary(id INT PRIMARY KEY,g INT,v LONGTEXT)')
        run.sql("INSERT INTO r10_ordinary VALUES(1,1,'hello'),(2,1,NULL),(3,2,REPEAT('世界',9000))")
        assert run.sql('SELECT g,COUNT(*),COUNT(v),SUM(id) FROM r10_ordinary GROUP BY g ORDER BY g') == ((1, 2, 1, 3), (2, 1, 1, 3))
        assert run.value('SELECT CHAR_LENGTH(v) FROM r10_ordinary WHERE id=3') == 18000
        assert run.value('SELECT RIGHT(v,4) FROM r10_ordinary WHERE id=3') == '世界世界'
        assert run.sql('SELECT a.id,b.id FROM r10_ordinary a JOIN r10_ordinary b ON a.g=b.g WHERE a.id=1 ORDER BY b.id') == ((1, 1), (1, 2))
        run.sql('START TRANSACTION')
        run.sql('DELETE FROM r10_ordinary WHERE id=1')
        run.sql('ROLLBACK')
        assert run.value('SELECT COUNT(*) FROM r10_ordinary') == 3
        run.sql('DROP TABLE r10_ordinary')
    run.case('ordinary/dml-join-aggregate-lob-and-rollback', relational_lob)


def main():
    import pymysql
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2883)
    parser.add_argument('--group', nargs='+', choices=['foundations', 'protocols', 'batch', 'ordinary'],
                        default=['foundations', 'protocols', 'batch', 'ordinary'])
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    connection = pymysql.connect(host='127.0.0.1', port=args.port, user='root', database='test',
                                 charset='utf8mb4', autocommit=True, read_timeout=180)
    report = {'kind': 'native SQL + independent local protocol fixture; supplemental tests',
              'groups': args.group, 'request_log': str(args.log), 'cases': []}
    run = Run(connection, args.log, report)
    report['version'] = run.value('SELECT VERSION()')
    try:
        for group in args.group:
            globals()[group](run)
    finally:
        report.update(passed=sum(c['passed'] for c in report['cases']),
                      failed=sum(not c['passed'] for c in report['cases']))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
        connection.close()
    print(json.dumps({k: report[k] for k in ['passed', 'failed']}))
    raise SystemExit(bool(report['failed']))


if __name__ == '__main__':
    main()
