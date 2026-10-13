"""Parity tests against the independent PyTorch implementation, not native formulas."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch

from neurogen.language.config import LanguageConfig
from neurogen.language.model import LanguageModel
from neurogen.language.native import NativeLanguageModel, library
from neurogen.language.runtime import Session


class NativeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            library()
        except ValueError:
            if os.environ.get('NEUROGEN_TEST_BACKENDS'):
                raise
            raise unittest.SkipTest('Build the native library to run native parity tests')
        cls.backends = os.environ.get('NEUROGEN_TEST_BACKENDS', 'native-cpu').split(',')

    def pair(self, backend, seed=7, columns=3, width=17):
        torch.manual_seed(seed)
        config = LanguageConfig(columns=columns, width=width, regions=2, ticks=3, idle_steps=1,
                                speech_width=16, mel_bins=8)
        reference = LanguageModel(config).eval()
        # Exercise nonuniform trained recurrent weights instead of just defaults.
        with torch.no_grad():
            for region in reference.regions:
                region.raw_weight.add_(torch.randn_like(region.raw_weight) * .3)
        native = NativeLanguageModel.from_model(copy.deepcopy(reference), backend)
        return reference, native

    def same(self, a, b, atol=3e-5):
        if isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.same(x, y, atol)
        else:
            torch.testing.assert_close(a, b, atol=atol, rtol=3e-5)

    def test_tokens_idle_batch_and_full_state(self):
        for backend in self.backends:
            for seed, columns, width in [(7, 3, 17), (11, 1, 1)]:
                with self.subTest(backend=backend, seed=seed), torch.no_grad():
                    ref, native = self.pair(backend, seed, columns, width)
                    a = b = None
                    for i in range(24):
                        token = torch.tensor([257 if i == 0 else i, 100 + i])
                        ra, rb = ref.step(token, a), native.step(token, b)
                        self.same(ra, rb)
                        a, b = ra[1], rb[1]
                    previous = b
                    original = tuple(tuple(x.clone() for x in r) for r in b)
                    for _ in range(8):
                        ra = ref.advance(torch.zeros(2, width), a, idle=True)
                        rb = native.advance(torch.zeros(2, width), b, idle=True)
                        self.same(ra, rb)
                        a, b = ra[1], rb[1]
                    # Previous caller-owned states must remain immutable.
                    self.same(previous, original)
                    self.assertTrue(any(bool(x.any()) for region in b for x in region))

    def test_session_cross_backend_resume_and_independent_sessions(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                ref, native = self.pair(backend)
                a, b = Session(ref), Session(native)
                a.hear_text('hello'); b.hear_text('hello')
                self.same(a.state, b.state)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'session.pt'
                    a.save(path, 'same-weights')
                    b.load(path, 'same-weights')
                    a.idle(4); b.idle(4)
                    self.same(a.state, b.state)
                    b.save(path, 'same-weights')
                    a.load(path, 'same-weights')
                    self.same(a.feed([42]), b.feed([42]))
                other = Session(native)
                before = tuple(tuple(x.clone() for x in r) for r in b.state)
                other.feed([10, 11, 12])
                self.same(b.state, before)

    def test_audio_and_sequence_frontends(self):
        for backend in self.backends:
            with self.subTest(backend=backend), torch.no_grad():
                ref, native = self.pair(backend)
                tokens = torch.tensor([[257, 65, 66]])
                self.same(ref(tokens), native(tokens))
                features = torch.rand(1, 3, 8)
                self.same(ref.listen(features), native.listen(features))
                ha = ref(tokens)[3]; hb = native(tokens)[3]
                self.same(ref.acoustic_decode(features, ha), native.acoustic_decode(features, hb))

    def test_shared_handle_threads(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                _, native = self.pair(backend)
                def run(token):
                    with torch.no_grad():
                        return native(torch.tensor([[257, token, token]]))
                expected = [run(10), run(20)]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    actual = list(pool.map(run, [10, 20]))
                self.same(expected, actual)

    def test_invalid_state_training_and_weight_reload(self):
        ref, native = self.pair('native-cpu')
        with self.assertRaisesRegex(ValueError, 'inference-only'):
            native.train()
        with self.assertRaises(ValueError):
            native.step(torch.tensor([-1]))
        state = native.initial(1)
        state[0][0][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            native.step(torch.tensor([1]), state)
        with torch.no_grad():
            native.head.bias.add_(1)
        with self.assertRaisesRegex(ValueError, 'weights changed'):
            native.step(torch.tensor([1]))
        native.load_state_dict(ref.state_dict())
        with torch.no_grad():
            self.same(ref.step(torch.tensor([1])), native.step(torch.tensor([1])))
        changed = ref.state_dict()
        changed['regions.0.mask'] = torch.zeros_like(changed['regions.0.mask'])
        with self.assertRaisesRegex(ValueError, 'architecture buffer'):
            native.load_state_dict(changed)

    def test_cli_stream_and_workbench_native_selection(self):
        import json
        import subprocess
        import sys
        import threading
        from dataclasses import asdict
        from neurogen.workbench.worker import Engine
        from neurogen.language.training import FORMAT, atomic_save
        ref, _ = self.pair('native-cpu', columns=1, width=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            atomic_save({'format': FORMAT, 'config': asdict(ref.config),
                         'model': ref.state_dict(), 'speech_trained': False}, path)
            result = subprocess.run([sys.executable, '-m', 'neurogen', 'language', 'stream',
                                     '--checkpoint', str(path), '--device', 'native-cpu',
                                     '--max-new-tokens', '2', '--temperature', '0'],
                                    input=json.dumps({'type': 'text', 'text': 'hi'}) + '\n',
                                    capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertTrue(any(e['type'] == 'turn_end' for e in events))
            self.assertFalse(any(e['type'] == 'error' for e in events))
            emitted = []
            worker = Engine(lambda event, **data: emitted.append((event, data)))
            worker.execute({'action': 'load', 'path': str(path), 'device': 'native-cpu'}, threading.Event())
            self.assertIsInstance(worker.session.model, NativeLanguageModel)
            worker.execute({'action': 'idle', 'steps': 2}, threading.Event())
            self.assertEqual(emitted[-1][0], 'activity')
            worker.unload()
