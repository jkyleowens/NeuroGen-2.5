"""Language tests are optional when only the biological simulator is installed."""
import importlib.util
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from dataclasses import asdict

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from neurogen.language.model import LanguageConfig, LanguageModel, BOS, EOS, PAD
    from neurogen.language.data import Corpus, record_tokens, disjoint
    from neurogen.language.audio import Features, StreamingFeatures, collapse_ctc, speech_objectives, write_wave, read_wave
    from neurogen.language.runtime import Session
    from neurogen.language.training import atomic_save, load_checkpoint, FORMAT
    from neurogen.language.cli import add_language_parser
    from neurogen.language.training import train


@unittest.skipUnless(HAS_TORCH, 'optional PyTorch language dependency not installed')
class LanguageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(21)
        self.model = LanguageModel(LanguageConfig.preset('tiny'))

    def test_device_diagnostics_distinguish_build_and_driver_failures(self):
        from unittest.mock import patch
        from neurogen.language.backend import device_for, doctor
        with patch('torch.version.cuda', None), patch('torch.version.hip', None), \
                patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(device_for('auto').type, 'cpu')
            with self.assertRaisesRegex(ValueError, 'CPU-only build'):
                device_for('cuda')
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaisesRegex(ValueError, 'CPU-only build'):
                doctor(argparse.Namespace(device='cuda', threads=1, preset='tiny'))
            report = json.loads(output.getvalue())
            self.assertFalse(report['diagnostic_passed'])
            self.assertEqual(report['python_executable'], sys.executable)
            self.assertIsNone(report['cuda_build'])
        with patch('torch.version.cuda', '12.8'), patch('torch.version.hip', None), \
                patch('torch.cuda.is_available', return_value=False), \
                patch('torch.cuda.init', side_effect=RuntimeError('driver unavailable')):
            with self.assertRaisesRegex(ValueError, 'driver unavailable'):
                device_for('cuda')
            with self.assertRaisesRegex(ValueError, 'not a HIP/ROCm'):
                device_for('rocm')
        with patch('torch.version.cuda', None), patch('torch.version.hip', '6.3'), \
                patch('torch.cuda.is_available', return_value=True):
            with self.assertRaisesRegex(ValueError, 'HIP/ROCm build'):
                device_for('cuda')
            self.assertEqual(device_for('rocm').type, 'cuda')

    def test_streaming_matches_sequence_and_has_causal_gradients(self):
        x = torch.tensor([[BOS, 104, 105, 33, EOS]])
        logits, state, active, _ = self.model(x)
        prefix = self.model(x[:, :3])[0]
        self.assertTrue(torch.equal(logits[:, :3], prefix))
        session, parts = Session(self.model), []
        state = None
        for value in x.unbind(1):
            result, state, _, _ = self.model.step(value, state)
            parts.append(result)
        self.assertTrue(torch.equal(logits, torch.stack(parts, 1)))
        logits.square().mean().backward()
        for region in self.model.regions:
            self.assertGreater(float(region.input.weight.grad.norm()), 0)
            self.assertGreater(float(region.raw_weight.grad.norm()), 0)
        self.assertGreater(float(active.detach()), 0)
        self.assertLessEqual(float(active.detach()), .5)

    def test_local_dale_signs_and_laminar_input(self):
        region = self.model.regions[0]
        effective = torch.nn.functional.softplus(region.raw_weight) * region.mask * region.sign
        self.assertTrue(bool((effective[..., region.sign < 0] <= 0).all()))
        self.assertTrue(bool((effective[..., region.sign > 0] >= 0).all()))
        self.assertTrue(bool((region.layer[region.input_ids] == 4).all()))
        self.assertTrue(bool((effective.diagonal(dim1=-2, dim2=-1) == 0).all()))
        _, state, _, _ = self.model(torch.randint(0, 256, (2, 64)))
        report = self.model.activity_report(state, 128)
        for index in range(2):
            self.assertGreater(report[f'region_{index}_excitatory'], 0)
            self.assertGreater(report[f'region_{index}_inhibitory'], 0)

    def test_dialogue_mask_corpus_and_overlap_rejection(self):
        ids, mask = record_tokens({'messages': [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'hi'}]})
        self.assertEqual([value for value, supervised in zip(ids, mask) if supervised], [104, 105, EOS])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'text.jsonl'
            path.write_text(json.dumps({'text': 'abcdef'}) + '\n')
            a, b = Corpus(path, root / 'a'), Corpus(path, root / 'b')
            with self.assertRaisesRegex(ValueError, 'identical'):
                disjoint(a, b)
            x, y = a.batch(2, 12, torch.Generator().manual_seed(1), 'cpu')
            self.assertEqual(x.shape, (2, 12))
            self.assertTrue(bool((y[:, 7:] == -100).all()))

    def test_causal_audio_chunk_equivalence_and_ctc_boundary(self):
        frontend = Features(64)
        for length in (1, 511, 512, 513, 832, 2107):
            samples = torch.randn(length) * .1
            stream = StreamingFeatures(frontend)
            chunks = [stream.push(samples[start:start + 97], final=start + 97 >= length) for start in range(0, length, 97)]
            actual = torch.cat(chunks)
            self.assertTrue(torch.allclose(actual, frontend(samples), atol=1e-6))
        a, previous = collapse_ctc([97, 97])
        b, _ = collapse_ctc([97, PAD, 97, 98], previous)
        self.assertEqual(a + b, [97, 97, 98])

    def test_speech_objectives_train_both_directions_and_vocoder(self):
        samples = .1 * torch.sin(torch.arange(4000) * .17)
        frontend = Features(64)
        features = frontend(samples).unsqueeze(0)
        ctc, mel, stop, _, _ = speech_objectives(self.model, features, 'a')
        (ctc * .1 + mel + stop).backward()
        for parameter in (self.model.audio_input.weight, self.model.speech_output.weight, self.model.speech_decoder.weight_ih_l0):
            self.assertGreater(float(parameter.grad.norm()), 0)
        with self.assertRaisesRegex(ValueError, 'CTC needs'):
            speech_objectives(self.model, features[:, :1], 'aa')
        audio = frontend.synthesize(features[0], iterations=2)
        self.assertTrue(bool(torch.isfinite(audio).all()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.wav'
            write_wave(path, audio)
            self.assertEqual(len(read_wave(path)), len(audio))

    def test_session_checkpoint_preserves_neural_sampling_and_audio_state(self):
        session = Session(self.model)
        session.hear_text('hello')
        session.hear_audio(torch.ones(173) * .01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.pt'
            session.save(path, 'test-model')
            restored = Session(self.model)
            restored.load(path, 'test-model')
            self.assertEqual(session.token(), restored.token())
            self.assertTrue(torch.equal(session.frontend.buffer, restored.frontend.buffer))
            with self.assertRaisesRegex(ValueError, 'different'):
                restored.load(path, 'wrong-model')

    def test_idle_updates_state_without_sampling_output(self):
        session = Session(self.model)
        previous = session.state[0][0].clone()
        random = session.random.get_state().clone()
        session.idle(3)
        self.assertFalse(torch.equal(previous, session.state[0][0]))
        self.assertTrue(torch.equal(random, session.random.get_state()))

    def test_loss_decreases_on_next_symbol_task(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01)
        x = torch.tensor([[BOS, 97, 98, 99, 97, 98, 99]])
        y = torch.tensor([[97, 98, 99, 97, 98, 99, EOS]])
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            logits = self.model(x)[0]
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0] * .6)

    def test_optimizer_resume_matches_uninterrupted_training(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train.txt').write_text('abcabcabcabc\n')
            (root / 'val.txt').write_text('bcabcabcabca\n')
            parser = argparse.ArgumentParser()
            add_language_parser(parser.add_subparsers(dest='command'))

            def run(name, steps, resume=None):
                arguments = ['language', 'train', '--preset', 'tiny', '--device', 'cpu',
                             '--train', str(root / 'train.txt'), '--validation', str(root / 'val.txt'),
                             '--steps', str(steps), '--batch-size', '1', '--sequence-length', '8',
                             '--accumulate', '1', '--eval-every', '2', '--eval-batches', '1',
                             '--output', str(root / name)]
                if resume:
                    arguments += ['--resume', str(resume)]
                with contextlib.redirect_stdout(io.StringIO()), patch('neurogen.language.training.charts'):
                    train(parser.parse_args(arguments))
                return root / name / 'checkpoint.pt'

            complete = run('complete', 4)
            partial = run('partial', 2)
            resumed = run('resumed', 4, partial)
            a = torch.load(complete, weights_only=True)
            b = torch.load(resumed, weights_only=True)
            self.assertEqual(a['step'], b['step'])
            for key, value in a['model'].items():
                self.assertTrue(torch.equal(value, b['model'][key]), key)
            self.assertTrue(torch.equal(a['sampler_rng'], b['sampler_rng']))

    def test_stream_cli_emits_text_and_asynchronous_wave_and_saves(self):
        # Fixed output verifies the protocol, not learned speech quality.
        with torch.no_grad():
            self.model.head.weight.zero_()
            self.model.head.bias.zero_()
            self.model.head.bias[97] = 10
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.pt'
            atomic_save({'format': FORMAT, 'config': asdict(self.model.config),
                         'model': self.model.state_dict(), 'speech_trained': True}, checkpoint)
            result = subprocess.run([sys.executable, '-m', 'neurogen', 'language', 'stream',
                '--checkpoint', str(checkpoint), '--device', 'cpu', '--temperature', '0',
                '--max-new-tokens', '4', '--voice-max-frames', '5', '--voice-output', str(root / 'voice'),
                '--save-session', str(root / 'session.pt')],
                input='[]\n' + json.dumps({'type': 'text', 'text': 'hello'}) + '\n',
                text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertIn('error', [e['type'] for e in events])
            self.assertEqual(''.join(e['text'] for e in events if e['type'] == 'text_delta'), 'aaaa')
            audio = next(e for e in events if e['type'] == 'audio_output')
            self.assertGreater(len(read_wave(audio['path'])), 0)
            self.assertTrue((root / 'session.pt').exists())

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'GPU unavailable')
    def test_gpu_forward_backward(self):
        model = self.model.cuda()
        loss = model(torch.tensor([[BOS, 97, EOS]], device='cuda'))[0].square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
