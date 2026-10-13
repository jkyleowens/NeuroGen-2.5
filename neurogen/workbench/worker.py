"""Isolated model process. JSONL commands/events keep GPU work off the Tk thread."""
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

from .configuration import inference_settings, estimate, save_json


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Engine:
    def __init__(self, emit):
        import torch
        self.torch, self.emit = torch, emit
        torch.set_num_threads(1)
        self.session, self.info, self.identity = None, None, None
        self.settings = inference_settings({})
        self.ticks = 0
        self.activity_known = True

    def require_model(self):
        if self.session is None:
            raise ValueError('Load a checkpoint or create a model first')
        return self.session

    def unload(self):
        self.session, self.info, self.identity = None, None, None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def execute(self, request, cancelled):
        from neurogen.language.model import LanguageModel, LanguageConfig, BOS, EOS, USER, ASSISTANT
        from neurogen.language.runtime import Session, synthesize_speech
        from neurogen.language.training import load_checkpoint, atomic_save, FORMAT
        from neurogen.language.backend import device_for, environment
        from neurogen.language.data import encode
        from neurogen.language.audio import read_wave, write_wave
        action, token = request['action'], request.get('id')

        def emit(event, **values):
            self.emit(event, id=token, **values)

        if action == 'unload':
            self.unload()
            emit('unloaded')
        elif action in ('load', 'create'):
            from neurogen.language.native import NATIVE_DEVICES, inference_model
            requested_device = request.get('device', 'auto')
            device = device_for('cpu' if requested_device in NATIVE_DEVICES else requested_device)
            seed = int(request.get('seed', 1))
            if seed < 0:
                raise ValueError('Seed must be nonnegative')
            if action == 'load':
                path = Path(request['path']).expanduser().resolve()
                before = path.stat()
                model, payload = load_checkpoint(path, device)
                identity = fingerprint(path)
                after = path.stat()
                if (before.st_mtime_ns, before.st_size, before.st_ino) != (after.st_mtime_ns, after.st_size, after.st_ino):
                    raise ValueError('Checkpoint changed while loading; try again after its writer finishes')
                details = {'path': str(path), 'step': payload.get('step', 0),
                           'speech_trained': bool(payload.get('speech_trained', False)),
                           'resume_available': bool(payload.get('optimizer')), 'settings': payload.get('settings', {}),
                           'data_hashes': payload.get('data_hashes', {}), 'data_source': payload.get('data_source', {})}
                del payload
            else:
                config = LanguageConfig(**request['config'])
                if estimate(asdict(config))['weights_mib'] > 2048:
                    raise ValueError('Interactive model creation is limited to 2 GiB of weights; use a smaller size')
                self.torch.manual_seed(seed)
                model = LanguageModel(config).to(device)
                identity = 'unsaved-' + uuid.uuid4().hex
                details = {'path': None, 'step': 0, 'speech_trained': False, 'resume_available': False, 'settings': {}}
            model = inference_model(model, requested_device)
            candidate = Session(model, seed)
            if cancelled.is_set():
                emit('cancelled')
                return
            self.session, self.identity = candidate, identity
            self.ticks = model.config.ticks * (1 + model.config.idle_steps)
            self.activity_known = True
            self.info = {**model.specification(), **details, 'environment': {**environment(device), 'device': getattr(model, 'native_backend', str(device))}}
            gc.collect()
            emit('loaded', info=self.info)
        elif action == 'settings':
            self.settings = inference_settings(request['settings'])
            emit('settings', settings=self.settings)
        elif action == 'reset':
            session = self.require_model()
            session.reset()
            self.ticks = session.model.config.ticks * (1 + session.model.config.idle_steps)
            self.activity_known = True
            emit('reset')
        elif action == 'idle':
            steps = int(request.get('steps', 1))
            self.require_model().idle(steps)
            self.ticks += steps * self.session.model.config.ticks
            emit('activity', populations=self.activity())
        elif action == 'chat':
            session = self.require_model()
            text = request['text']
            ids = encode(text)
            if not ids or len(ids) > 8192:
                raise ValueError('Prompt must contain 1..8192 UTF-8 bytes')
            settings = inference_settings(request.get('settings', self.settings))
            self.settings = settings
            mode = request.get('mode', 'dialogue')
            if mode not in ('dialogue', 'continuation'):
                raise ValueError('Mode must be dialogue or continuation')
            ids = [USER] + ids + [EOS, ASSISTANT] if mode == 'dialogue' else ids
            session.decoder.reset()
            began = time.perf_counter()
            generated, first_token, ended = 0, None, False
            tick_per_token = session.model.config.ticks * (1 + session.model.config.idle_steps)
            for value in ids:
                if cancelled.is_set():
                    break
                session.feed([value])
                self.ticks += tick_per_token
            if not cancelled.is_set():
                for _ in range(settings['max_new_tokens']):
                    if cancelled.is_set():
                        break
                    delta, ended = session.token(settings['temperature'], settings['top_k'])
                    generated += 1
                    self.ticks += tick_per_token
                    if first_token is None:
                        first_token = time.perf_counter() - began
                    if delta:
                        emit('delta', text=delta)
                    if ended:
                        break
            if not ended:
                tail = session.finish()
                self.ticks += tick_per_token
                if tail:
                    emit('delta', text=tail)
            elapsed = time.perf_counter() - began
            emit('turn_end', reason='cancelled' if cancelled.is_set() else ('eos' if ended else 'length'),
                 tokens=generated, seconds=elapsed, tokens_s=generated / max(elapsed, 1e-9),
                 first_token_seconds=first_token, populations=self.activity())
        elif action == 'score':
            session = self.require_model()
            ids = encode(request['text'])
            if not 1 <= len(ids) <= 8192:
                raise ValueError('Scoring needs 1..8192 UTF-8 bytes')
            state, outputs = None, []
            with self.torch.no_grad():
                for value in [BOS] + ids:
                    if cancelled.is_set():
                        emit('cancelled')
                        return
                    logits, state, _, _ = session.model.step(self.torch.tensor([value], device=session.device), state)
                    outputs.append(logits)
                logits = self.torch.cat(outputs)
                targets = self.torch.tensor(ids + [EOS], device=session.device)
                loss = float(self.torch.nn.functional.cross_entropy(logits, targets))
                accuracy = float((logits.argmax(-1) == targets).float().mean())
            emit('score', loss=loss, accuracy=accuracy, perplexity=math.exp(min(20, loss)), targets=len(ids) + 1)
        elif action == 'save_model':
            session = self.require_model()
            path = Path(request['path']).expanduser().resolve()
            atomic_save({'format': FORMAT, 'kind': 'weights_only', 'config': asdict(session.model.config),
                         'model': session.model.state_dict(), 'step': self.info['step'],
                         'speech_trained': self.info['speech_trained']}, path)
            self.identity = fingerprint(path)
            self.info.update(path=str(path), resume_available=False)
            emit('model_saved', path=str(path), info=self.info)
        elif action == 'save_session':
            session = self.require_model()
            if not self.info.get('path'):
                raise ValueError('Save this new model checkpoint before saving its session')
            path = Path(request['path']).expanduser().resolve()
            session.save(path, self.identity)
            save_json(path.with_suffix(path.suffix + '.workbench.json'),
                      {'checkpoint_id': self.identity, 'ticks': self.ticks, 'activity_known': self.activity_known,
                       'conversation': request.get('conversation', [])})
            emit('session_saved', path=str(path))
        elif action == 'load_session':
            session = self.require_model()
            path = Path(request['path']).expanduser().resolve()
            candidate = Session(session.model)
            candidate.load(path, self.identity)
            sidecar = path.with_suffix(path.suffix + '.workbench.json')
            view = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
            if view and view.get('checkpoint_id') != self.identity:
                raise ValueError('Session conversation belongs to a different checkpoint')
            conversation = view.get('conversation', [])
            if not isinstance(conversation, list) or any(not isinstance(row, dict) or not isinstance(row.get('text'), str) for row in conversation):
                raise ValueError('Invalid conversation sidecar')
            self.session, self.ticks = candidate, int(view.get('ticks', 0))
            self.activity_known = bool(view.get('activity_known', bool(self.ticks)))
            emit('session_loaded', conversation=conversation)
        elif action == 'listen':
            session = self.require_model()
            if not self.info['speech_trained']:
                raise ValueError('This checkpoint has not received paired speech training')
            samples = read_wave(request['path'], max_seconds=60)
            session.reset_audio()
            transcript, began = '', time.perf_counter()
            for start in range(0, len(samples), 320):
                if cancelled.is_set():
                    break
                delta = session.hear_audio(samples[start:start + 320], final=start + 320 >= len(samples))
                transcript += delta
                if delta:
                    emit('transcript_delta', text=delta)
            session.reset_audio()
            emit('transcript', text=transcript, cancelled=cancelled.is_set(),
                 real_time_factor=(time.perf_counter() - began) / (len(samples) / 16000))
        elif action == 'speak':
            session = self.require_model()
            if not self.info['speech_trained']:
                raise ValueError('This checkpoint has not received paired speech training')
            output = Path(request['path']).expanduser().resolve()
            result = synthesize_speech(session.model, request['text'], int(request.get('max_frames', 300)), cancelled)
            if result is None:
                emit('cancelled')
            else:
                samples, report = result
                output.parent.mkdir(parents=True, exist_ok=True)
                write_wave(output, samples)
                emit('audio', path=str(output), **report)
        else:
            raise ValueError(f'Unknown worker action: {action}')

    def activity(self):
        if self.session is None or not self.ticks or not self.activity_known:
            return {}
        return self.session.model.activity_report(self.session.state, self.ticks)


def main():
    output_lock = threading.Lock()

    def emit(event, **values):
        with output_lock:
            print(json.dumps({'event': event, **values}, allow_nan=False), flush=True)

    try:
        engine = Engine(emit)
    except ImportError as error:
        emit('fatal', message=f'Language workbench needs PyTorch: {error}. See LANGUAGE.md for installation.')
        return
    inbox, pending, lock = queue.Queue(maxsize=32), {}, threading.Lock()

    def reader():
        for line in sys.stdin:
            try:
                if len(line) > 2_000_000:
                    raise ValueError('Command too large')
                request = json.loads(line)
                if not isinstance(request, dict) or 'action' not in request:
                    raise ValueError('Command must be a JSON object with an action')
                with lock:
                    if request['action'] in ('cancel', 'shutdown'):
                        for cancellation in pending.values():
                            cancellation.set()
                    if request['action'] == 'cancel':
                        continue
                    cancellation = threading.Event()
                    pending[request.get('id')] = cancellation
                inbox.put((request, cancellation))
            except (ValueError, TypeError) as error:
                emit('error', message=str(error))
        inbox.put(({'action': 'shutdown'}, threading.Event()))

    threading.Thread(target=reader, daemon=True).start()
    emit('ready')
    last_activity = time.monotonic()
    while True:
        timeout = 1 / engine.settings['idle_hz'] if engine.session and engine.settings['continuous'] else .2
        try:
            request, cancelled = inbox.get(timeout=timeout)
        except queue.Empty:
            if engine.session and engine.settings['continuous']:
                try:
                    engine.session.idle()
                    engine.ticks += engine.session.model.config.ticks
                    if time.monotonic() - last_activity >= 1:
                        emit('activity', populations=engine.activity())
                        last_activity = time.monotonic()
                except Exception as error:
                    engine.settings['continuous'] = False
                    emit('error', message=f'Idle updates paused: {error}')
            continue
        if request['action'] == 'shutdown':
            engine.unload()
            break
        token = request.get('id')
        emit('busy', id=token, action=request['action'])
        try:
            if cancelled.is_set():
                emit('cancelled', id=token)
            else:
                engine.execute(request, cancelled)
        except Exception as error:
            emit('error', id=token, action=request['action'], message=f'{type(error).__name__}: {error}')
        finally:
            with lock:
                pending.pop(token, None)
            emit('done', id=token)
    emit('closed')


if __name__ == '__main__':
    main()
