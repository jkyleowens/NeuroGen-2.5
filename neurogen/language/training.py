"""Supervised byte/dialogue + CTC + acoustic training with reproducible resumption."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.nn import functional as F

from .audio import Features, SpeechCorpus, speech_objectives, edit_distance
from .backend import device_for, environment
from .data import Corpus, disjoint
from .corpus import text_sources
from .model import LanguageConfig, LanguageModel

FORMAT = 'neurogen-language-v2'


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parents[1].rglob('*.py')):
        digest.update(str(path.relative_to(Path(__file__).resolve().parents[1])).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Unsupported language checkpoint; biological checkpoints are separate')
    model = LanguageModel(LanguageConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['model'], strict=True)
    return model, payload


def charts(history, output):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    groups = [('Text loss', ('train_text_loss', 'val_text_loss')),
              ('Token accuracy', ('val_accuracy',)),
              ('Spikes / unit / internal tick (not Hz)', ('activity', 'excitatory_activity', 'inhibitory_activity')),
              ('Speech losses', ('val_ctc_loss', 'val_mel_loss', 'val_stop_loss')),
              ('Recognition error rate', ('val_cer', 'val_wer')),
              ('Supervised tokens / wall second', ('supervised_tokens_s',))]
    for ax, (title, keys) in zip(axes.flat, groups):
        for key in keys:
            rows = [r for r in history if key in r]
            if rows:
                ax.plot([r['step'] for r in rows], [r[key] for r in rows], label=key)
        ax.set(title=title, xlabel='Optimizer step')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'Not measured in this run', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(output / 'learning.png', dpi=130)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, corpus, speech, args, device, frontend):
    model.eval()
    generator = torch.Generator().manual_seed(1907)
    loss, correct, count, activity = 0., 0, 0, 0.
    populations = {}
    for _ in range(args.eval_batches):
        x, y = corpus.batch(args.batch_size, args.sequence_length, generator, device)
        logits, state, active, _ = model(x)
        for key, value in model.activity_report(state, x.shape[1] * model.config.ticks * (1 + model.config.idle_steps)).items():
            populations[key] = populations.get(key, 0.) + value / args.eval_batches
        valid = y != -100
        loss += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum'))
        correct += int(((logits.argmax(-1) == y) & valid).sum())
        count += int(valid.sum())
        activity += float(active)
    result = {'val_text_loss': loss / count, 'val_accuracy': correct / count,
              'val_perplexity': math.exp(min(20, loss / count)), 'activity': activity / args.eval_batches}
    result.update(populations)
    for population in ('excitatory', 'inhibitory'):
        result[population + '_activity'] = sum(value for key, value in populations.items() if key.endswith('_' + population)) / model.config.regions
    if speech:
        ctc, mel, stop, chars, words, char_errors, word_errors = 0., 0., 0., 0, 0, 0, 0
        n = min(args.eval_batches, len(speech.rows))
        for i in range(n):
            features, reference = speech.example(i, frontend, device)
            lc, lm, ls, _, transcript = speech_objectives(model, features, reference)
            ctc += float(lc)
            mel += float(lm)
            stop += float(ls)
            chars += len(reference)
            words += len(reference.split())
            char_errors += edit_distance(reference, transcript)
            word_errors += edit_distance(reference.split(), transcript.split())
        result.update(val_ctc_loss=ctc / n, val_mel_loss=mel / n, val_stop_loss=stop / n,
                      val_cer=char_errors / max(1, chars), val_wer=word_errors / max(1, words))
    model.train()
    return result


def train(args):
    for key in ('steps', 'batch_size', 'sequence_length', 'accumulate', 'eval_every', 'eval_batches', 'threads', 'max_audio_seconds'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Learning rate must be positive and finite')
    if bool(args.speech_train) != bool(args.speech_validation):
        raise ValueError('Supply both speech training and speech validation manifests')
    if bool(args.train) != bool(args.validation):
        raise ValueError('Supply both --train and --validation, or neither for automatic SlimPajama')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    output = Path(args.output or ('runs/language-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; use a new directory when resuming')
    output.mkdir(parents=True, exist_ok=True)
    settings = {key: getattr(args, key) for key in ('batch_size', 'sequence_length', 'accumulate', 'learning_rate',
                'seed', 'ctc_weight', 'mel_weight', 'stop_weight', 'max_audio_seconds')}
    for key in ('ctc_weight', 'mel_weight', 'stop_weight'):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError('Speech loss weights must be nonnegative and finite')
    config = LanguageConfig.preset(args.preset)
    if args.config:
        config = LanguageConfig(**(asdict(config) | json.loads(Path(args.config).read_text())))
    payload = None
    initial_speech_trained = False
    if args.resume:
        model, payload = load_checkpoint(args.resume, device)
        if not payload.get('optimizer') or 'settings' not in payload:
            raise ValueError('This checkpoint contains model weights only; use --initialize-from to start training')
        initial_speech_trained = bool(payload.get('speech_trained'))
        if args.config and model.config != config:
            raise ValueError('Resume configuration differs from checkpoint')
    elif args.initialize_from:
        model, initial = load_checkpoint(args.initialize_from, device)
        initial_speech_trained = bool(initial.get('speech_trained'))
        if args.config and model.config != config:
            raise ValueError('Weight initialization supplies architecture; use that configuration or start a fresh model')
    else:
        model = LanguageModel(config).to(device)
    train_path, validation_path, data_source = text_sources(args, output, payload)
    train_data = Corpus(train_path, output / 'data/train')
    validation = Corpus(validation_path, output / 'data/validation')
    disjoint(train_data, validation)
    speech = SpeechCorpus(args.speech_train, args.max_audio_seconds) if args.speech_train else None
    speech_val = SpeechCorpus(args.speech_validation, args.max_audio_seconds) if speech else None
    if speech:
        disjoint(speech, speech_val)
    data_hashes = {'train': train_data.sha256, 'validation': validation.sha256,
                   'speech_train': speech.sha256 if speech else None,
                   'speech_validation': speech_val.sha256 if speech_val else None}
    if payload and (payload['data_hashes'] != data_hashes or payload['settings'] != settings):
        raise ValueError('Resume requires identical corpora and training settings')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sampler = torch.Generator().manual_seed(args.seed + 1)
    history, start = [], 0
    if payload:
        optimizer.load_state_dict(payload['optimizer'])
        sampler.set_state(payload['sampler_rng'])
        torch.set_rng_state(payload['torch_rng'])
        if device.type == 'cuda' and payload.get('gpu_rng'):
            torch.cuda.set_rng_state_all(payload['gpu_rng'])
        start, history = payload['step'], payload['history']
    if args.steps <= start:
        raise ValueError('--steps is the total desired optimizer steps and must exceed the checkpoint step')
    frontend = Features(model.config.mel_bins, device)
    metadata = {'format': FORMAT, **model.specification(), 'environment': environment(device),
                'source_sha256': source_hash(), 'data_hashes': data_hashes, 'data_source': data_source, 'settings': settings,
                'speech_supervision_enabled': bool(speech), 'resume': str(args.resume) if args.resume else None,
                'initialize_from': str(args.initialize_from) if args.initialize_from else None,
                'limitations': 'Supervised research model; no pretrained language or speech knowledge. Idle updates are not proof of reasoning.'}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'output': str(output), **metadata}), flush=True)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    budget = args.memory_budget_mib
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Memory budget must be positive and finite')
    # Conservative planning estimate only; runtime allocations are measured separately.
    estimated = model.specification()['fp32_adam_parameter_state_mib'] + (
        args.batch_size * args.sequence_length * model.config.regions * model.config.columns * 64 *
        model.config.ticks * (1 + model.config.idle_steps) * 4 * 24 / 1024**2)
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        if estimated > min(budget, free / 1024**2 * .8):
            raise ValueError(f'Estimated training memory {estimated:.0f} MiB exceeds available budget; reduce batch/sequence/size')

    previous_handler = signal.signal(signal.SIGINT, request_stop)

    def save(step):
        atomic_save({'format': FORMAT, 'config': asdict(model.config), 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'step': step, 'settings': settings,
                     'data_hashes': data_hashes, 'data_source': data_source, 'sampler_rng': sampler.get_state(),
                     'torch_rng': torch.get_rng_state(),
                     'gpu_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                     'history': history,
                     'speech_trained': initial_speech_trained or (bool(speech) and step > 0 and args.ctc_weight > 0 and args.mel_weight > 0 and args.stop_weight > 0),
                     'source_sha256': source_hash()}, output / 'checkpoint.pt')

    def record(row):
        history.append(row)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        charts(history, output)

    try:
        if history:
            (output / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
        record({'step': start, **evaluate(model, validation, speech_val, args, device, frontend)})
        save(start)
        began = time.perf_counter()
        interval_loss, interval_tokens, interval_steps = 0., 0, 0
        for step in range(start + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses, token_count = [], 0
            for _ in range(args.accumulate):
                x, y = train_data.batch(args.batch_size, args.sequence_length, sampler, device)
                logits, _, _, _ = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Non-finite language loss; last complete checkpoint retained')
                (loss / args.accumulate).backward()
                losses.append(float(loss.detach()))
                token_count += int((y != -100).sum())
            if speech:
                index = int(torch.randint(len(speech.rows), (), generator=sampler))
                features, text = speech.example(index, frontend, device)
                ctc, mel, stopping, _, _ = speech_objectives(model, features, text)
                speech_loss = args.ctc_weight * ctc + args.mel_weight * mel + args.stop_weight * stopping
                if not bool(torch.isfinite(speech_loss)):
                    raise ValueError('Non-finite speech loss; last complete checkpoint retained')
                speech_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            interval_loss += sum(losses) / len(losses)
            interval_tokens += token_count
            interval_steps += 1
            peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0.
            if step % args.eval_every == 0 or step == args.steps or stop_requested or peak > budget:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - began
                row = {'step': step, 'train_text_loss': interval_loss / interval_steps,
                       'supervised_tokens_s': interval_tokens / elapsed, 'gradient_norm': float(norm),
                       'peak_gpu_allocated_mib': peak,
                       **evaluate(model, validation, speech_val, args, device, frontend)}
                record(row)
                save(step)
                interval_loss, interval_tokens, interval_steps = 0., 0, 0
                began = time.perf_counter()
            if stop_requested or peak > budget:
                print('Saved at completed optimizer step; interrupt or GPU memory budget reached.', flush=True)
                break
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    (output / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>NeuroGen language training</title>'
        '<h1>Language and speech training</h1><p>Spiking activity is per internal tick, not biological Hz. '
        'Speech charts require paired audio training. Recognition error can exceed 100%.</p>'
        '<img src="learning.png" style="max-width:100%" alt="Loss, accuracy, spikes, speech error and throughput">'
        '<p><a href="metrics.jsonl">Raw metrics</a> · <a href="run.json">Configuration and provenance</a></p>')
    return output
