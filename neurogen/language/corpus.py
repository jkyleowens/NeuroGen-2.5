"""Create bounded, provenance-tracked SlimPajama subsets for individual runs."""
import hashlib
import inspect
import json
from pathlib import Path
import secrets
import shutil

DEFAULT_DATASET = 'gmongaras/SlimPajama-627B_Reupload'


def progress(phase, **values):
    print(json.dumps({'event': 'corpus_progress', 'phase': phase, **values}), flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_revision(dataset_id, revision):
    from huggingface_hub import HfApi
    return HfApi().dataset_info(dataset_id, revision=revision, expand=['sha'], timeout=30).sha


def stream_split(dataset_id, revision, split, seed, buffer_size, max_bytes):
    from datasets import load_dataset
    # The default reupload is Parquet. Avoid materializing whole row groups as
    # Python batches or reading unused metadata columns on consumer hardware.
    reader = {'columns': ['text'], 'batch_size': 128} if dataset_id == DEFAULT_DATASET else {}
    dataset = load_dataset(dataset_id, revision=revision, split=split, streaming=True, **reader)
    # Truncate before buffering so a shuffle buffer cannot retain huge documents.
    def bounded(record):
        text = record.get('text')
        if not isinstance(text, str):
            raise ValueError('SlimPajama records must have a string text field')
        return {'text': text.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore').strip()}
    options = {'seed': seed, 'buffer_size': buffer_size}
    # Datasets 5 adds concurrent shard readers by default. A bounded subset
    # needs one reader; concurrent prefetch can outlive a cancelled iterator.
    if 'max_buffer_input_shards' in inspect.signature(dataset.shuffle).parameters:
        options['max_buffer_input_shards'] = 1
    return iter(dataset.select_columns(['text']).map(bounded).shuffle(**options))


def prepare_slimpajama(output, *, dataset_id=DEFAULT_DATASET, revision='main', seed=None,
                      train_documents=2000, validation_documents=128,
                      max_document_bytes=8192, shuffle_buffer=128):
    for name, value in (('train_documents', train_documents), ('validation_documents', validation_documents),
                        ('max_document_bytes', max_document_bytes), ('shuffle_buffer', shuffle_buffer)):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if seed is not None and not 0 <= seed < 2**32:
        raise ValueError('Dataset seed must be between 0 and 4294967295')
    if not dataset_id.strip() or not revision.strip():
        raise ValueError('SlimPajama dataset ID and revision cannot be empty')
    try:
        import datasets  # noqa: F401
    except ImportError as error:
        raise ValueError("Automatic SlimPajama needs Datasets. Run: python -m pip install -e '.[language]'") from error
    directory = Path(output) / 'corpus'
    directory.mkdir(parents=True, exist_ok=True)
    paths = {split: directory / (split + '.jsonl') for split in ('train', 'validation')}
    if any(path.exists() for path in paths.values()) or (directory / 'manifest.json').exists():
        raise ValueError('Generated corpus already exists; choose a new run directory')
    seed = secrets.randbits(32) if seed is None else seed
    progress('resolving', message='Preparing a new SlimPajama subset…', dataset_id=dataset_id, seed=seed)
    seen, counts, sizes, hashes = set(), {}, {}, {}
    partials = [path.with_suffix('.jsonl.part') for path in paths.values()]
    try:
        commit = dataset_revision(dataset_id, revision)
        # Reserve held-out documents first, then exclude them from training.
        for split, target in (('validation', validation_documents), ('train', train_documents)):
            count, size, scanned = 0, 0, 0
            digest = hashlib.sha256()
            progress('downloading', split=split, documents=0, target=target)
            rows = stream_split(dataset_id, commit, split, seed, shuffle_buffer, max_document_bytes)
            try:
                with paths[split].with_suffix('.jsonl.part').open('wb') as stream:
                    for record in rows:
                        scanned += 1
                        # Also enforce the bound here for alternative/test stream implementations.
                        text = record['text'].encode('utf-8')[:max_document_bytes].decode('utf-8', errors='ignore').strip()
                        key = hashlib.sha256(text.encode('utf-8')).digest()
                        if text and key not in seen:
                            seen.add(key)
                            line = (json.dumps({'text': text}, ensure_ascii=False) + '\n').encode('utf-8')
                            stream.write(line)
                            digest.update(line)
                            count += 1
                            size += len(line)
                            if count % 100 == 0 or count == target:
                                progress('downloading', split=split, documents=count, target=target, bytes=size)
                            if count == target:
                                break
                        if scanned >= max(10000, target * 100):
                            break
            finally:
                close = getattr(rows, 'close', None)
                if close:
                    close()
            if count != target:
                raise ValueError(f'Only found {count}/{target} unique usable {split} documents; reduce the sample size')
            counts[split], sizes[split], hashes[split] = count, size, digest.hexdigest()
        manifest = {'kind': 'slimpajama', 'dataset_id': dataset_id, 'revision': commit,
                    'requested_revision': revision, 'seed': seed, 'documents': counts, 'jsonl_bytes': sizes,
                    'max_document_bytes': max_document_bytes, 'shuffle_buffer': shuffle_buffer,
                    'sha256': hashes, 'files': {split: str(path.relative_to(output)) for split, path in paths.items()}}
        for partial, path in zip(partials, paths.values()):
            partial.replace(path)
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        progress('ready', message='SlimPajama training and validation files are ready.', documents=counts)
        return paths['train'], paths['validation'], manifest
    except KeyboardInterrupt:
        progress('cancelled', message='Dataset preparation stopped; no model training has started.')
        raise
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        raise ValueError(f'SlimPajama preparation failed for {dataset_id} ({type(error).__name__}'
                         f'{", HTTP " + str(status) if status else ""}). Check network access and dataset availability; '
                         'authenticate with hf auth login or HF_TOKEN if required.') from error
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)


def text_sources(args, output, payload=None):
    """Local inputs remain supported; optimizer resumption never resamples data."""
    if bool(args.train) != bool(args.validation):
        raise ValueError('Supply both --train and --validation, or neither for automatic SlimPajama')
    if args.train:
        if args.data_source == 'slimpajama':
            raise ValueError('Use --data-source files with local files, or omit --train/--validation for SlimPajama')
        paths = [Path(value).expanduser().resolve() for value in (args.train, args.validation)]
        return *paths, {'kind': 'files', 'train': str(paths[0]), 'validation': str(paths[1])}
    if payload is not None:
        previous = payload.get('data_source', {})
        if previous.get('kind') == 'slimpajama':
            paths = {split: Path(args.resume).resolve().parent / 'corpus' / (split + '.jsonl')
                     for split in ('train', 'validation')}
            for split, path in paths.items():
                if not path.is_file() or file_hash(path) != previous['sha256'][split]:
                    raise ValueError('Resume needs the unchanged corpus/ folder beside the checkpoint; '
                                     'copy it with the checkpoint or supply the original --train/--validation files')
            directory = Path(output) / 'corpus'
            directory.mkdir(parents=True, exist_ok=True)
            for split, path in paths.items():
                shutil.copyfile(path, directory / path.name)
            provenance = {**previous, 'reused_from': str(Path(args.resume).resolve())}
            (directory / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
            progress('ready', message='Reusing the checkpoint’s saved SlimPajama subset; no download required.')
            return directory / 'train.jsonl', directory / 'validation.jsonl', provenance
        if previous.get('kind') == 'files':
            return Path(previous['train']), Path(previous['validation']), previous
        raise ValueError('This older checkpoint needs its original --train and --validation files to resume')
    if args.data_source == 'files':
        raise ValueError('--data-source files requires --train and --validation')
    return prepare_slimpajama(output, dataset_id=args.dataset_id, revision=args.dataset_revision,
                             seed=args.dataset_seed, train_documents=args.train_documents,
                             validation_documents=args.validation_documents,
                             max_document_bytes=args.max_document_bytes, shuffle_buffer=args.shuffle_buffer)
