#!/usr/bin/env python
# encoding: utf-8

"""
Offline unit tests for the experiment pipeline (no MongoDB, Ollama, OpenAI or cluster needed).

    cd experiment
    python -m unittest tests/test_units.py -v
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEDULER_DIR = os.path.join(EXPERIMENT_DIR, 'python-llm-toolkit', 'sbatch_scheduler')
sys.path.insert(0, os.path.join(EXPERIMENT_DIR, 'framework'))
sys.path.insert(0, EXPERIMENT_DIR)
sys.path.insert(0, SCHEDULER_DIR)

os.environ.setdefault('LOG_DIR', os.path.join(tempfile.gettempdir(), 'recommendation_test_logs'))

import httpx  # noqa: E402

import models as models_config  # noqa: E402
import progress_utils  # noqa: E402
import recommendation_experiment as rx  # noqa: E402
import recommendation_evaluator as ev  # noqa: E402
from llm_response import LlmResponse  # noqa: E402
from recommendation_utils import ResultWriter, RESULT_FIELDNAMES  # noqa: E402
import check_squeue  # noqa: E402
import sbatch_scheduler  # noqa: E402
import utils as scheduler_utils  # noqa: E402


def make_submission(i, domain='Books', selftext=''):
    return {
        'id': f'sub{i}',
        'title': f'[IIL] Title {i} [WEWIL?]',
        'selftext': selftext,
        'link_flair_text': domain,
        'final_recommendations': [{'title': f'Book {i}-{k}', 'ISBN': f'978-{i}{k}'} for k in range(6)],
    }


def ok_response(payload):
    return LlmResponse(ok=True, finish_reason='stop', message=json.dumps(payload),
                       message_token_count=10, prompt_token_count=100, duration_seconds=1.0)


class TestModels(unittest.TestCase):
    def test_setups(self):
        self.assertEqual(len(models_config.get_ollama_models()), 20)
        self.assertEqual(len(models_config.get_models_for_setup("tiny_and_small_models")), 8)
        self.assertEqual(models_config.get_models_for_setup('openai_models'),
                         ['openai:gpt-5-nano', 'openai:gpt-5-mini', 'openai:gpt-5.2'])
        self.assertEqual(len(models_config.get_all()), 23)
        with self.assertRaises(ValueError):
            models_config.get_models_for_setup('tiny')

    def test_routing(self):
        self.assertTrue(models_config.is_proprietary_model('openai:gpt-5-mini'))
        self.assertFalse(models_config.is_proprietary_model('gpt-oss:20b'))
        self.assertEqual(models_config.get_model_version('openai:gpt-5.2'), 'gpt-5.2')
        self.assertEqual(models_config.model_slug('qwen3:32b-q4_K_M'), 'qwen3_32b-q4_K_M')


class TestPromptConstruction(unittest.TestCase):
    def test_input_modes(self):
        sub = make_submission(1, selftext='More details')
        self.assertEqual(rx.build_submission_text(sub, 'title'), sub['title'])
        self.assertEqual(rx.build_submission_text(sub, 'title_selftext'), f"{sub['title']}\n\nMore details")
        sub['selftext'] = '[removed]'
        self.assertEqual(rx.build_submission_text(sub, 'title_selftext'), sub['title'])

    def test_prompts_exist_for_all_domains(self):
        for prompt_type in rx.PROMPT_TYPES:
            for domain in rx.DOMAIN_TO_PROMPT:
                self.assertIn('<submission_text>', rx.load_prompt(prompt_type, domain))

    def test_few_shot_deterministic_and_excludes_self(self):
        subs = [make_submission(i) for i in range(20)]
        a = rx.FewShotSampler(subs, seed=42).sample(subs[0])
        b = rx.FewShotSampler(list(reversed(subs)), seed=42).sample(subs[0])
        c = rx.FewShotSampler(subs, seed=7).sample(subs[0])
        self.assertEqual([e['id'] for e in a], [e['id'] for e in b])  # independent of order
        self.assertNotEqual([e['id'] for e in a], [e['id'] for e in c])  # depends on seed
        self.assertEqual(len(a), 3)
        self.assertNotIn('sub0', [e['id'] for e in a])

    def test_few_shot_prompt_fills_placeholders(self):
        subs = [make_submission(i) for i in range(5)]
        examples = rx.FewShotSampler(subs, seed=1).sample(subs[0])
        prompt = rx.format_few_shot_prompt(rx.load_prompt('few_shot', 'Books'), 'REQUEST', examples)
        self.assertNotIn('<random_example', prompt)
        self.assertNotIn('<submission_text>', prompt)
        self.assertIn('REQUEST', prompt)
        self.assertIn('Example 3:', prompt)


class TestParsing(unittest.TestCase):
    def test_expected_structure(self):
        r = rx.parse_recommendation_response(ok_response({'recommendations': [{'title': 'A'}, {'title': 'a '}, {'title': 'B'}]}))
        self.assertTrue(r['valid_json'] and r['valid_structure'])
        self.assertEqual((r['nr_recs'], r['nr_recs_unique']), (3, 2))

    def test_other_key_is_extracted_but_flagged(self):
        r = rx.parse_recommendation_response(ok_response({'books': [{'title': 'A'}]}))
        self.assertTrue(r['valid_json'])
        self.assertFalse(r['valid_structure'])
        self.assertEqual(r['nr_recs'], 1)

    def test_non_string_title_does_not_break(self):
        r = rx.parse_recommendation_response(ok_response({'recommendations': [{'title': 1984}, {'title': None}]}))
        self.assertIsNone(r['error'])
        self.assertEqual(r['nr_recs_unique'], 1)

    def test_code_fences_and_invalid_json(self):
        fenced = LlmResponse(ok=True, message='```json\n{"recommendations": []}\n```')
        self.assertTrue(rx.parse_recommendation_response(fenced)['valid_json'])
        broken = LlmResponse(ok=True, message='{"recommendations": [')
        r = rx.parse_recommendation_response(broken)
        self.assertFalse(r['valid_json'])
        self.assertIn('JSON decode error', r['error'])

    def test_failed_response(self):
        r = rx.parse_recommendation_response(LlmResponse(ok=False, finish_reason='api_error', error='400 bad'))
        self.assertEqual(r['error'], '400 bad')


class TestResultWriter(unittest.TestCase):
    def test_append_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'm', 'zero_shot_results.csv')
            with ResultWriter(path) as w:
                w.write({'submission_id': 'a', 'error': 'x'})          # partial (error) row first
                w.write({'submission_id': 'b', 'raw_response': 'line1\nline2'})
            with ResultWriter(path) as w:
                self.assertEqual(w.done_ids, {'a', 'b'})
                w.write({'submission_id': 'c'})
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, RESULT_FIELDNAMES)
                self.assertEqual([r['submission_id'] for r in reader], ['a', 'b', 'c'])

    def test_unknown_column_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ResultWriter(os.path.join(tmp, 'r.csv')) as w:
                with self.assertRaises(ValueError):
                    w.write({'submission_id': 'a', 'bogus': 1})

    def test_old_format_file_is_moved_aside(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'r.csv')
            with open(path, 'w') as f:
                f.write('submission_id,domain\nx,Books\n')
            with ResultWriter(path) as w:
                self.assertEqual(w.done_ids, set())
            self.assertTrue(any(n.endswith('.old_format') for n in os.listdir(tmp)))


class TestProcessSubmissions(unittest.TestCase):
    """End-to-end over the main loop with the LLM calls mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.subs = [make_submission(i, domain=d) for i, d in enumerate(['Books', 'Film', 'TV', 'Music', 'Games'] * 2)]
        self.progress = os.path.join(self.tmp.name, 'run', 'progress_m.json')
        patches = [
            mock.patch.object(rx, 'ensure_model_available', return_value=True),
            mock.patch.object(rx.ollama_client, 'get_server_version', return_value='0.0-test'),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def run_once(self, generate):
        with mock.patch.object(rx, 'generate_recommendations', side_effect=generate):
            return rx.process_submissions(self.subs, self.subs, ['llama3.2:3b-instruct-q4_K_M'],
                                          ['zero_shot', 'few_shot'], self.progress, seed=42)

    def test_full_run_then_resume_after_infrastructure_error(self):
        calls = {'n': 0}

        def flaky(model, prompt):
            calls['n'] += 1
            if calls['n'] == 4:
                raise rx.InfrastructureError('tunnel down')
            return ok_response({'recommendations': [{'title': 'X'}]})

        with self.assertRaises(rx.InfrastructureError):
            self.run_once(flaky)
        with open(self.progress) as f:
            progress = json.load(f)
        self.assertEqual(progress['progress_entries'][0]['prompts']['0'], 'failed')
        self.assertEqual(progress['setup']['runConfig']['seed'], 42)
        self.assertEqual(progress['setup']['runConfig']['ollamaServerVersion'], '0.0-test')

        csv_path = os.path.join(self.tmp.name, 'run', 'llama3.2_3b-instruct-q4_K_M', 'zero_shot_results.csv')
        with open(csv_path, newline='', encoding='utf-8') as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)  # rows before the failure are kept

        calls['n'] = 100  # no more failures
        self.assertEqual(self.run_once(flaky), rx.EXIT_OK)
        for prompt_type in ('zero_shot', 'few_shot'):
            path = csv_path.replace('zero_shot', prompt_type)
            with open(path, newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(sorted(r['submission_id'] for r in rows), sorted(s['id'] for s in self.subs))
        with open(self.progress) as f:
            self.assertEqual(json.load(f)['progress_entries'][0]['status'], 'finished')
        self.assertEqual(calls['n'] - 100, 20 - 3)  # only the missing submissions were called

    def test_few_shot_examples_recorded(self):
        self.run_once(lambda model, prompt: ok_response({'recommendations': []}))
        path = os.path.join(self.tmp.name, 'run', 'llama3.2_3b-instruct-q4_K_M', 'few_shot_results.csv')
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(len(json.loads(r['few_shot_example_ids'])) == 1 for r in rows))  # 2 per domain -> 1 candidate

    def test_model_not_available(self):
        with mock.patch.object(rx, 'ensure_model_available', return_value=False):
            code = self.run_once(lambda model, prompt: ok_response({}))
        self.assertEqual(code, rx.EXIT_MODEL_NOT_AVAILABLE)
        with open(self.progress) as f:
            entry = json.load(f)['progress_entries'][0]
        self.assertEqual(entry['status'], 'not available')


class TestRetry(unittest.TestCase):
    def test_transport_errors_retried_then_raised(self):
        with mock.patch.object(rx.ollama_client, 'generate', side_effect=httpx.ConnectError('refused')) as gen, \
                mock.patch.object(rx.time, 'sleep'):
            with self.assertRaises(rx.InfrastructureError):
                rx.generate_recommendations('llama3.2:3b-instruct-q4_K_M', 'p')
        self.assertEqual(gen.call_count, rx.config.LLM_MAX_RETRIES)

    def test_recovers_after_transient_error(self):
        effects = [httpx.ReadTimeout('slow'), ok_response({'recommendations': []})]
        with mock.patch.object(rx.ollama_client, 'generate', side_effect=effects), mock.patch.object(rx.time, 'sleep'):
            self.assertTrue(rx.generate_recommendations('llama3.2:3b-instruct-q4_K_M', 'p').ok)

    def test_openai_models_are_routed_to_openai(self):
        with mock.patch.object(rx.openai_client, 'generate', return_value=ok_response({})) as gen:
            rx.generate_recommendations('openai:gpt-5-mini', 'p')
        self.assertEqual(gen.call_args.kwargs['model'], 'gpt-5-mini')


class TestProgressUtils(unittest.TestCase):
    def test_status_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'p.json')
            progress_utils.create_initial_progress_structure(['m'], ['zero_shot', 'few_shot'], path)
            progress_utils.set_model_prompt_running('m', 0, path)
            progress_utils.set_model_prompt_finished('m', 0, path)
            entry = progress_utils.read_progress(path)[0]
            self.assertEqual(entry['status'], 'running')
            progress_utils.set_model_prompt_finished('m', 1, path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data['progress_entries'][0]['status'], 'finished')
            self.assertIsNotNone(data['setup']['endTimestamp'])
            self.assertFalse(os.path.exists(path + '.tmp'))


class TestScheduler(unittest.TestCase):
    def test_parse_remote_host(self):
        self.assertEqual(scheduler_utils.parse_remote_host('u@h'), ('u@h', None))
        self.assertEqual(scheduler_utils.parse_remote_host('u@h:2222'), ('u@h', '2222'))
        self.assertEqual(scheduler_utils.ssh_prefix('u@h:2222')[-3:], ['-p', '2222', 'u@h'])

    def test_replace_parameters(self):
        with open(os.path.join(SCHEDULER_DIR, 'ollama_serve.sh')) as f:
            script = f.read()
        script = sbatch_scheduler.replace_parameter_in_script(script, 'PORT_NUMBER', 28021)
        script = sbatch_scheduler.replace_parameter_in_script(script, '--gpus-per-node', 2)
        script = sbatch_scheduler.replace_parameter_in_script(script, '--gpus', 2)
        self.assertIn('PORT_NUMBER=28021', script)
        self.assertIn('#SBATCH --gpus-per-node=2', script)
        self.assertIn('#SBATCH --gpus=2', script)
        self.assertIn('OLLAMA_HOST=127.0.0.1:$PORT_NUMBER', script)

    def test_port_validation(self):
        self.assertEqual(sbatch_scheduler.port_number('28021'), 28021)
        with self.assertRaises(Exception):
            sbatch_scheduler.port_number('71195')

    def _squeue(self, stdout='', returncode=0):
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr='')

    def test_check_squeue_parses_format(self):
        with mock.patch.object(check_squeue.subprocess, 'run', return_value=self._squeue('"123|hungryapeman|RUNNING|1:02|worker12"\n')):
            job = check_squeue.check_squeue(123, 'hungryapeman', 'hungryapeman@nvcluster.tugraz.at')
        self.assertEqual(job['state'], 'RUNNING')
        self.assertEqual(job['node'], 'worker12')

    def test_check_squeue_job_gone(self):
        with mock.patch.object(check_squeue.subprocess, 'run', return_value=self._squeue('', 1)):
            self.assertIsNone(check_squeue.check_squeue(123, 'u', 'u@h'))

    def test_monitor_job_gives_up_when_job_disappears(self):
        with mock.patch.object(sbatch_scheduler, 'check_squeue', return_value=None), \
                mock.patch.object(sbatch_scheduler.time, 'sleep'):
            self.assertEqual(sbatch_scheduler.monitor_job(1, 'u', 'u@h', max_not_found=3), (False, None))

    def test_monitor_job_waits_for_running(self):
        states = [
            {'jobid': '1', 'user': 'u', 'state': 'PENDING', 'time': '0:00', 'node': '(Resources)'},
            {'jobid': '1', 'user': 'u', 'state': 'RUNNING', 'time': '0:01', 'node': 'worker3'},
        ]
        with mock.patch.object(sbatch_scheduler, 'check_squeue', side_effect=states), \
                mock.patch.object(sbatch_scheduler.time, 'sleep'):
            self.assertEqual(sbatch_scheduler.monitor_job(1, 'u', 'u@h'), (True, 'worker3'))

    def test_monitor_job_terminal_state(self):
        job = {'jobid': '1', 'user': 'u', 'state': 'FAILED', 'time': '0:00', 'node': ''}
        with mock.patch.object(sbatch_scheduler, 'check_squeue', return_value=job):
            self.assertEqual(sbatch_scheduler.monitor_job(1, 'u', 'u@h'), (False, None))


class TestRunModelsQuality(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(EXPERIMENT_DIR, 'setup', 'scripts'))
        import run_models
        self.rm = run_models
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(run_models, 'OUTPUT_DIR', self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def _write(self, prompt_type, valid_flags):
        path = os.path.join(self.tmp.name, 'gpt-oss_20b', f'{prompt_type}_results.csv')
        with ResultWriter(path) as w:
            for i, ok in enumerate(valid_flags):
                w.write({'submission_id': str(i), 'valid_json': ok, 'response_truncated': not ok})

    def test_mostly_invalid_answers_give_warn(self):
        self._write('zero_shot', [False, False, False])  # like the gpt-oss smoke test with 1000 tokens
        self._write('few_shot', [True, True, True])
        status = self.rm.quality_status(self.rm.answer_quality('gpt-oss:20b'))
        self.assertTrue(status.startswith('WARN'))
        self.assertIn('zero_shot: 0% valid JSON, 100% truncated (n=3)', status)

    def test_good_answers_are_ok(self):
        self._write('zero_shot', [True, True, False])
        self.assertTrue(self.rm.quality_status(self.rm.answer_quality('gpt-oss:20b')).startswith('OK'))


class TestEvaluator(unittest.TestCase):
    def _write(self, path, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with ResultWriter(path) as w:
            for r in rows:
                w.write(r)

    def _row(self, sid, recs, ts, valid=True):
        truth = [{'title': 'Dune', 'ISBN': '978-0441013593'}, {'title': 'Hyperion'}] + [{'title': f'T{i}'} for i in range(4)]
        return {'submission_id': sid, 'domain': 'Books', 'model': 'm', 'prompt_type': 'zero_shot',
                'valid_json': valid, 'valid_structure': valid, 'nr_recs': len(recs),
                'llm_recommendations': json.dumps(recs), 'ground_truth': json.dumps(truth), 'timestamp': ts}

    def test_metrics(self):
        row = self._row('s', [{'title': 'X', 'ISBN': '9780441013593'}, {'title': 'hyperion'}, {'title': 'Nope'}], 't')
        result = ev.evaluate_row(row)
        self.assertEqual(result['id_match_count'], 1)
        self.assertAlmostEqual(result['exact_match_precision'], 2 / 3)
        self.assertAlmostEqual(result['exact_match_recall'], 2 / 6)

    def test_skips_old_outputs_and_dedups_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(os.path.join(tmp, 'run_a', 'm', 'zero_shot_results.csv'),
                        [self._row('s1', [{'title': 'Nope'}], '2026-01-01T00:00:00')])
            self._write(os.path.join(tmp, 'run_b', 'm', 'zero_shot_results.csv'),
                        [self._row('s1', [{'title': 'Dune'}], '2026-02-01T00:00:00')])
            # a previous evaluation output inside the results dir must be ignored
            self._write(os.path.join(tmp, 'evaluated', 'run_a', 'm', 'zero_shot_results.csv'), [self._row('s9', [], 't')])
            self._write(os.path.join(tmp, 'old_eval', 'x_results.csv'), [self._row('s8', [], 't')])
            open(os.path.join(tmp, 'old_eval', ev.OUTPUT_MARKER), 'w').close()

            files = ev.find_result_csvs(tmp, os.path.join(tmp, 'evaluated_new'))
            self.assertEqual(len(files), 2)

            out = os.path.join(tmp, 'evaluated_new')
            os.makedirs(out)
            dfs = [ev.evaluate_csv(f, os.path.join(out, os.path.relpath(f, tmp)),
                                   source_run=os.path.relpath(f, tmp).split(os.sep)[0]) for f in files]
            ev.aggregate_results(dfs, out)
            import pandas as pd
            all_eval = pd.read_csv(os.path.join(out, 'all_evaluated.csv'))
            self.assertEqual(len(all_eval), 1)
            self.assertEqual(all_eval.iloc[0]['source_run'], 'run_b')  # most recent run kept
            adherence = pd.read_csv(os.path.join(out, 'format_adherence_by_model_prompttype.csv'))
            self.assertEqual(adherence.iloc[0]['rows'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
