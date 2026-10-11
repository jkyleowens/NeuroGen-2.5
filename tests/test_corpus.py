"""Dataset preparation is tested offline; real Hub downloads are explicit smoke tests."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neurogen.language.corpus import DEFAULT_DATASET, prepare_slimpajama, stream_split, text_sources

HAS_DATASETS = importlib.util.find_spec('datasets') is not None
HAS_TORCH = importlib.util.find_spec('torch') is not None


def records(dataset, revision, split, seed, buffer_size, max_bytes):
    texts = (['held out', 'held out', 'A training', 'B training', 'C training'] if split == 'train'
             else ['held out', 'validation other'])
    return iter({'text': text} for text in texts)


@unittest.skipUnless(HAS_DATASETS, 'optional Datasets dependency not installed')
class CorpusPreparationTests(unittest.TestCase):
    def prepare(self, path, **kwargs):
        with patch('neurogen.language.corpus.dataset_revision', return_value='pinned-commit'), \
                patch('neurogen.language.corpus.stream_split', side_effect=records), \
                contextlib.redirect_stdout(io.StringIO()):
            return prepare_slimpajama(path, train_documents=2, validation_documents=1, **kwargs)

    def test_bounded_unique_held_out_text_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation, manifest = self.prepare(root, seed=17, max_document_bytes=9)
            training = [json.loads(line)['text'] for line in train.read_text().splitlines()]
            held_out = [json.loads(line)['text'] for line in validation.read_text().splitlines()]
            self.assertEqual(training, ['A training'[:9], 'B training'[:9]])
            self.assertFalse(set(training) & set(held_out))
            self.assertEqual(len(set(training)), 2)
            self.assertTrue(all(len(text.encode()) <= 9 for text in training + held_out))
            self.assertEqual(manifest['revision'], 'pinned-commit')
            self.assertEqual(manifest['seed'], 17)
            self.assertEqual(manifest['sha256']['train'], hashlib.sha256(train.read_bytes()).hexdigest())
            self.assertEqual(manifest, json.loads((root / 'corpus/manifest.json').read_text()))

    def test_fresh_seed_each_run_and_explicit_seed_reproduction(self):
        with tempfile.TemporaryDirectory() as directory, patch('neurogen.language.corpus.secrets.randbits', side_effect=[11, 22]):
            root = Path(directory)
            a = self.prepare(root / 'a')
            b = self.prepare(root / 'b')
            self.assertEqual((a[2]['seed'], b[2]['seed']), (11, 22))
            c = self.prepare(root / 'c', seed=11)
            self.assertEqual(a[0].read_bytes(), c[0].read_bytes())

    def test_real_stream_pipeline_is_bounded_and_seeded(self):
        from datasets import IterableDataset
        def source(offsets):
            for offset in offsets:
                for i in range(offset * 30, (offset + 1) * 30):
                    yield {'text': f'{i:02d} ' + 'é' * 100, 'meta': {'ignored': True}}
        with tempfile.TemporaryDirectory() as directory, patch('datasets.config.HF_DATASETS_CACHE', Path(directory)):
            dataset = IterableDataset.from_generator(source, gen_kwargs={'offsets': [0, 1, 2]})
        self.assertEqual(dataset.num_shards, 3)
        def sample(seed):
            with patch('datasets.load_dataset', return_value=dataset) as load:
                rows = list(stream_split('repository', 'commit', 'train', seed, 8, 9))
                load.assert_called_once_with('repository', revision='commit', split='train', streaming=True)
            self.assertTrue(all(len(row['text'].encode()) <= 9 for row in rows))
            self.assertTrue(all(set(row) == {'text'} for row in rows))
            return rows
        a = sample(1)
        self.assertEqual(a, sample(1))
        self.assertNotEqual(a, sample(2))
        with patch('datasets.load_dataset', return_value=dataset) as load:
            rows = stream_split(DEFAULT_DATASET, 'commit', 'train', 1, 8, 9)
            load.assert_called_once_with(DEFAULT_DATASET, revision='commit', split='train', streaming=True,
                                         columns=['text'], batch_size=128)
            next(rows)
            rows.close()

    def test_cancelled_preparation_cleans_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def interrupted(*args):
                yield {'text': 'one document'}
                raise KeyboardInterrupt
            with patch('neurogen.language.corpus.dataset_revision', return_value='commit'), \
                    patch('neurogen.language.corpus.stream_split', side_effect=interrupted), \
                    contextlib.redirect_stdout(io.StringIO()) as logs:
                with self.assertRaises(KeyboardInterrupt):
                    prepare_slimpajama(root, seed=1, train_documents=2, validation_documents=2)
            self.assertIn('"phase": "cancelled"', logs.getvalue())
            self.assertEqual(list((root / 'corpus').iterdir()), [])

    def test_unicode_bound_and_failed_preparation_leaves_no_complete_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def unicode_rows(dataset, revision, split, seed, buffer_size, max_bytes):
                return iter([{'text': ('é' if split == 'train' else 'ü') * 10}])
            with patch('neurogen.language.corpus.dataset_revision', return_value='commit'), \
                    patch('neurogen.language.corpus.stream_split', side_effect=unicode_rows), \
                    contextlib.redirect_stdout(io.StringIO()):
                train, _, _ = prepare_slimpajama(root / 'good', seed=1, train_documents=1,
                                               validation_documents=1, max_document_bytes=9)
                self.assertEqual(json.loads(train.read_text())['text'], 'é' * 4)
                with self.assertRaisesRegex(ValueError, 'unique usable'):
                    prepare_slimpajama(root / 'bad', seed=1, train_documents=2, validation_documents=1)
            self.assertEqual(list((root / 'bad/corpus').iterdir()), [])

    def test_resume_reuses_portable_corpus_offline_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, _, manifest = self.prepare(root / 'original', seed=3)
            args = argparse.Namespace(train=None, validation=None, data_source=None,
                                      resume=str(root / 'original/checkpoint.pt'))
            with patch('neurogen.language.corpus.prepare_slimpajama', side_effect=AssertionError('must stay offline')), \
                    contextlib.redirect_stdout(io.StringIO()):
                copied, _, source = text_sources(args, root / 'resumed', {'data_source': manifest})
                self.assertEqual(copied.read_bytes(), train.read_bytes())
                self.assertEqual(source['seed'], 3)
                train.write_text('{"text":"modified"}\n')
                with self.assertRaisesRegex(ValueError, 'unchanged corpus'):
                    text_sources(args, root / 'tampered', {'data_source': manifest})

    def test_incomplete_local_pair_fails_without_network(self):
        args = argparse.Namespace(train='train.txt', validation=None, data_source=None)
        with self.assertRaisesRegex(ValueError, 'both --train and --validation'):
            text_sources(args, Path('unused'))

    @unittest.skipUnless(HAS_TORCH, 'optional torch dependency not installed')
    def test_automatic_training_and_offline_resume_match_uninterrupted(self):
        import torch
        from neurogen.language.cli import add_language_parser
        from neurogen.language.training import train
        parser = argparse.ArgumentParser()
        add_language_parser(parser.add_subparsers(dest='command'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(name, steps, resume=None):
                arguments = ['language', 'train', '--device', 'cpu', '--preset', 'tiny', '--dataset-seed', '5',
                    '--train-documents', '2', '--validation-documents', '1', '--steps', str(steps),
                    '--batch-size', '1', '--sequence-length', '8', '--accumulate', '1', '--eval-every', '1',
                    '--eval-batches', '1', '--output', str(root / name)]
                if resume:
                    arguments += ['--resume', str(resume)]
                with contextlib.redirect_stdout(io.StringIO()), patch('neurogen.language.training.charts'):
                    train(parser.parse_args(arguments))
                return root / name / 'checkpoint.pt'
            with patch('neurogen.language.corpus.dataset_revision', return_value='commit'), \
                    patch('neurogen.language.corpus.stream_split', side_effect=records):
                complete = run('complete', 2)
                partial = run('partial', 1)
            with patch('neurogen.language.corpus.stream_split', side_effect=AssertionError('resume must not download')):
                resumed = run('resumed', 2, partial)
            a, b = (torch.load(path, weights_only=True) for path in (complete, resumed))
            self.assertEqual(b['data_source']['kind'], 'slimpajama')
            self.assertEqual(b['data_hashes'], a['data_hashes'])
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
            self.assertTrue(torch.equal(a['sampler_rng'], b['sampler_rng']))


if __name__ == '__main__':
    unittest.main()
