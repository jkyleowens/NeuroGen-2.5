"""Workbench backend tests run without a desktop, microphone, or GPU."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest

from neurogen.workbench.configuration import estimate, inference_settings, parse_config
from neurogen.workbench.processes import JobProcess, ModelProcess
from neurogen.language.config import LanguageConfig

HAS_TORCH = importlib.util.find_spec('torch') is not None


class ConfigurationTests(unittest.TestCase):
    def test_invalid_json_architecture_and_sampling(self):
        with self.assertRaises(ValueError):
            parse_config('[]')
        with self.assertRaises(ValueError):
            parse_config('{"columns": 0}')
        with self.assertRaises(ValueError):
            parse_config('{"unknown": 1}')
        for settings in ({'temperature': float('nan')}, {'top_k': 0}, {'max_new_tokens': -1}, {'idle_hz': 0}):
            with self.assertRaises(ValueError):
                inference_settings(settings)
        self.assertEqual(inference_settings({'temperature': 0})['temperature'], 0)

    @unittest.skipUnless(HAS_TORCH, 'optional torch not installed')
    def test_size_estimate_matches_real_model(self):
        from neurogen.language.model import LanguageModel
        for name in ('tiny', 'laptop', 'bc250'):
            config = LanguageConfig.preset(name)
            self.assertEqual(estimate(asdict(config))['parameters'], LanguageModel(config).specification()['parameters'])


@unittest.skipUnless(HAS_TORCH, 'optional torch not installed')
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.worker = ModelProcess(self.root)
        self.wait_for('ready')

    def tearDown(self):
        self.worker.stop()
        try:
            self.worker.process.wait(timeout=15)
        except Exception:
            self.worker.process.kill()
            self.worker.process.wait(timeout=5)
        for stream in (self.worker.process.stdin, self.worker.process.stdout, self.worker.process.stderr):
            stream.close()
        self.directory.cleanup()

    def wait_for(self, kind, timeout=25):
        end = time.monotonic() + timeout
        seen = []
        while time.monotonic() < end:
            try:
                event = self.worker.events.get(timeout=max(.01, end - time.monotonic()))
            except queue.Empty:
                break
            seen.append(event)
            if event['event'] == kind:
                return event, seen
            if event['event'] == 'fatal':
                self.fail(event['message'])
        self.fail(f'Timed out waiting for {kind}: {seen}')

    def action(self, action, **values):
        token = self.worker.send(action, **values)
        events = []
        end = time.monotonic() + 25
        while time.monotonic() < end:
            event = self.worker.events.get(timeout=max(.01, end - time.monotonic()))
            events.append(event)
            if event['event'] == 'done' and event.get('id') == token:
                return events
        self.fail('Worker action timed out')

    def create(self):
        events = self.action('create', config=asdict(LanguageConfig.preset('tiny')), device='cpu', seed=1)
        self.assertIn('loaded', [event['event'] for event in events])

    def test_create_save_load_chat_score_and_session_replay(self):
        self.create()
        checkpoint = self.root / 'new-model.pt'
        self.action('save_model', path=str(checkpoint))
        self.assertTrue(checkpoint.is_file())
        settings = {'temperature': .7, 'top_k': 8, 'max_new_tokens': 8}
        session = self.root / 'session.pt'
        conversation = [{'role': 'user', 'text': 'hello'}]
        self.action('save_session', path=str(session), conversation=conversation)
        a = self.action('chat', text='abc', mode='continuation', settings=settings)
        self.assertIn('turn_end', [event['event'] for event in a])
        scores = self.action('score', text='abc')
        self.assertIn('score', [event['event'] for event in scores])
        self.action('unload')
        loaded = self.action('load', path=str(checkpoint), device='cpu')
        self.assertFalse(next(event for event in loaded if event['event'] == 'loaded')['info']['resume_available'])
        restored = self.action('load_session', path=str(session))
        self.assertEqual(next(event for event in restored if event['event'] == 'session_loaded')['conversation'], conversation)
        b = self.action('chat', text='abc', mode='continuation', settings=settings)
        self.assertEqual(''.join(e['text'] for e in a if e['event'] == 'delta'),
                         ''.join(e['text'] for e in b if e['event'] == 'delta'))

    def test_error_does_not_discard_loaded_model_and_cancellation_is_responsive(self):
        self.create()
        failed = self.action('load', path=str(self.root / 'missing.pt'), device='cpu')
        self.assertIn('error', [e['event'] for e in failed])
        self.worker.send('chat', text='hello', settings={'max_new_tokens': 8192, 'temperature': 0})
        self.wait_for('busy')
        self.worker.send('cancel')
        _, events = self.wait_for('done')
        self.assertTrue(any(e['event'] in ('turn_end', 'cancelled') for e in events))
        reset = self.action('reset')
        self.assertIn('reset', [e['event'] for e in reset])
        no_speech = self.action('listen', path='unused.wav')
        self.assertIn('paired speech', next(e['message'] for e in no_speech if e['event'] == 'error'))

    def test_weights_can_initialize_training_and_job_logs_are_saved(self):
        self.create()
        checkpoint = self.root / 'model.pt'
        self.action('save_model', path=str(checkpoint))
        self.action('unload')
        (self.root / 'train.txt').write_text('abcabcabc\n')
        (self.root / 'val.txt').write_text('bcabcabca\n')
        job = JobProcess()
        output = self.root / 'run'
        args = ['language', 'train', '--initialize-from', str(checkpoint), '--device', 'cpu',
                '--train', str(self.root / 'train.txt'), '--validation', str(self.root / 'val.txt'),
                '--steps', '1', '--batch-size', '1', '--sequence-length', '8', '--accumulate', '1',
                '--eval-every', '1', '--eval-batches', '1', '--output', str(output)]
        job.start('train', args, self.root, self.root / 'job.log', str(output))
        try:
            result = job.process.wait(timeout=45)
            self.assertEqual(result, 0, (self.root / 'job.log').read_text())
            self.assertTrue((output / 'checkpoint.pt').is_file())
            self.assertTrue((output / 'learning.png').is_file())
            metadata = json.loads((output / 'run.json').read_text())
            self.assertEqual(metadata['config'], asdict(LanguageConfig.preset('tiny')))
            self.assertEqual(metadata['initialize_from'], str(checkpoint))
        finally:
            if job.running:
                job.stop(force=True)
                job.process.wait(timeout=5)
            job.process.stdout.close()


if __name__ == '__main__':
    unittest.main()
