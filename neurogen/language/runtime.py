"""Persistent language state and bounded, interruptible text/PCM streaming."""
import base64
import codecs
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .audio import Features, StreamingFeatures, read_wave, write_wave, collapse_ctc
from .backend import device_for
from .data import encode
from .model import PAD, BOS, EOS, USER, ASSISTANT
from .training import load_checkpoint, atomic_save


class Session:
    def __init__(self, model, seed=1):
        self.model = model.eval()
        self.device = model.embedding.weight.device
        self.random = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self):
        self.state = None
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.reset_audio()
        self.logits = self.feed([BOS])

    def reset_audio(self):
        self.audio_state, self.previous_ctc = None, PAD
        self.audio_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.frontend = StreamingFeatures(Features(self.model.config.mel_bins, self.device))

    @torch.no_grad()
    def feed(self, tokens):
        for value in tokens:
            logits, self.state, _, _ = self.model.step(torch.tensor([value], device=self.device), self.state)
            self.logits = logits[0]
        return self.logits

    def hear_text(self, text):
        self.decoder.reset()
        return self.feed([USER] + encode(text) + [EOS, ASSISTANT])

    @torch.no_grad()
    def idle(self, steps=1):
        if not 0 <= steps <= 100:
            raise ValueError('Idle steps must be between 0 and 100')
        for _ in range(steps):
            h, self.state, _ = self.model.advance(self.model.embedding.weight.new_zeros((1, self.model.config.width)), self.state, idle=True)
            self.logits = self.model.head(h)[0]

    @torch.no_grad()
    def token(self, temperature=.7, top_k=40):
        if not np.isfinite(temperature) or temperature < 0 or not 1 <= top_k <= 257:
            raise ValueError('Temperature must be finite/nonnegative; top_k must be 1..257')
        logits = self.logits.clone()
        logits[256:] = -torch.inf
        logits[EOS] = self.logits[EOS]
        if temperature == 0:
            value = int(logits.argmax())
        else:
            values, indices = torch.topk(logits / temperature, top_k)
            choice = torch.multinomial(values.softmax(-1).cpu(), 1, generator=self.random)
            value = int(indices[int(choice)])
        self.feed([value])
        if value == EOS:
            return self.decoder.decode(b'', final=True), True
        return self.decoder.decode(bytes([value])), False

    def finish(self):
        text = self.decoder.decode(b'', final=True)
        self.decoder.reset()
        self.feed([EOS])
        return text

    @torch.no_grad()
    def hear_audio(self, samples, final=False):
        frames = self.frontend.push(samples, final=final)
        result = ''
        if len(frames):
            logits, self.audio_state, _ = self.model.listen(frames.unsqueeze(0), self.audio_state)
            ids, self.previous_ctc = collapse_ctc(logits[0, :, :PAD + 1].argmax(-1).tolist(), self.previous_ctc)
            result = self.audio_decoder.decode(bytes(ids))
        if final:
            result += self.audio_decoder.decode(b'', final=True)
            self.reset_audio()
        return result

    def save(self, path, checkpoint_id):
        atomic_save({'format': 'neurogen-session-v2', 'checkpoint_id': checkpoint_id,
                     'state': self.state, 'logits': self.logits, 'random': self.random.get_state(),
                     'decoder': (list(self.decoder.getstate()[0]), self.decoder.getstate()[1]), 'audio_state': self.audio_state,
                     'previous_ctc': self.previous_ctc,
                     'audio_decoder': (list(self.audio_decoder.getstate()[0]), self.audio_decoder.getstate()[1]),
                     'audio_buffer': self.frontend.buffer, 'audio_emitted': self.frontend.emitted}, path)

    def load(self, path, checkpoint_id):
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get('format') != 'neurogen-session-v2' or payload.get('checkpoint_id') != checkpoint_id:
            raise ValueError('Session belongs to a different model checkpoint')
        self.state, self.logits = payload['state'], payload['logits']
        self.random.set_state(payload['random'].cpu())
        self.decoder.setstate((bytes(payload['decoder'][0]), payload['decoder'][1]))
        self.audio_state, self.previous_ctc = payload['audio_state'], payload['previous_ctc']
        self.audio_decoder.setstate((bytes(payload['audio_decoder'][0]), payload['audio_decoder'][1]))
        self.frontend.buffer, self.frontend.emitted = payload['audio_buffer'], payload['audio_emitted']


def load_runtime(args):
    torch.set_num_threads(args.threads)
    from .native import NATIVE_DEVICES, inference_model
    device = device_for('cpu' if args.device in NATIVE_DEVICES else args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    return inference_model(model, args.device), payload


def listen(args):
    model, payload = load_runtime(args)
    if not payload.get('speech_trained'):
        raise ValueError('Checkpoint has not been trained with paired speech')
    samples = read_wave(args.wav, args.max_audio_seconds)
    session = Session(model)
    began, transcript = time.perf_counter(), ''
    for start in range(0, len(samples), 320):
        chunk = session.hear_audio(samples[start:start + 320], final=start + 320 >= len(samples))
        transcript += chunk
        if chunk:
            print(json.dumps({'type': 'transcript_delta', 'text': chunk}), flush=True)
    print(json.dumps({'type': 'transcript', 'text': transcript,
                      'real_time_factor': (time.perf_counter() - began) / (len(samples) / 16000)}))


@torch.no_grad()
def synthesize_speech(model, text, max_frames, cancelled=None):
    if not text or len(encode(text)) > 2048 or not 1 <= max_frames <= 3000:
        raise ValueError('Speech needs 1..2048 text bytes and max_frames in 1..3000')
    started = time.perf_counter()
    ids = torch.tensor([[BOS] + encode(text) + [EOS]], device=model.embedding.weight.device)
    _, _, _, encoded = model(ids)
    frame, hidden, frames = encoded.new_zeros((1, 1, model.config.mel_bins)), None, []
    stopped = False
    for index in range(max_frames):
        if cancelled is not None and cancelled.is_set():
            return None
        frame, stop, hidden = model.acoustic_decode(frame, encoded, hidden)
        frames.append(frame[0, 0])
        if index >= 4 and float(stop.sigmoid()) > .5:
            stopped = True
            break
    # CPU vocoder avoids depending on GPU FFT/pseudoinverse support.
    samples = Features(model.config.mel_bins).synthesize(torch.stack(frames).cpu())
    if cancelled is not None and cancelled.is_set():
        return None
    return samples, {'seconds': len(samples) / 16000, 'stop_predicted': stopped,
                     'real_time_factor': (time.perf_counter() - started) / (len(samples) / 16000)}


def speak(args):
    model, payload = load_runtime(args)
    if not payload.get('speech_trained'):
        raise ValueError('Checkpoint has not been trained with paired speech')
    path = Path(args.output)
    if path.exists():
        raise ValueError('Speech output already exists')
    samples, report = synthesize_speech(model, args.text, args.max_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wave(path, samples)
    print(json.dumps({'output': str(path), **report}))


def stream(args):
    """JSONL protocol; a reader thread lets new input interrupt byte generation.

    Audio chunks are PCM16/16k/mono, max one second. State advances during silence.
    Timing is best-effort and reported, not promised to run faster than wall time.
    """
    import hashlib
    if not 0 < args.idle_hz <= 100 or not 1 <= args.max_new_tokens <= 8192:
        raise ValueError('idle_hz must be 0..100 and max_new_tokens 1..8192')
    if not 1 <= args.voice_max_frames <= 3000:
        raise ValueError('voice_max_frames must be between 1 and 3000')
    model, payload = load_runtime(args)
    if args.voice_output and not payload.get('speech_trained'):
        raise ValueError('--voice-output requires a checkpoint trained with paired speech')
    if not np.isfinite(args.temperature) or args.temperature < 0 or not 1 <= args.top_k <= 257:
        raise ValueError('Temperature must be finite/nonnegative; top_k must be 1..257')
    session = Session(model, args.seed)
    digest = hashlib.sha256()
    with Path(args.checkpoint).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    checkpoint_id = digest.hexdigest()
    if args.resume_session:
        session.load(args.resume_session, checkpoint_id)
    incoming = queue.Queue(maxsize=32)

    def reader():
        for line in sys.stdin:
            if args.action == 'chat':
                command = line.strip()
                line = json.dumps({'type': command[1:]}) if command in ('/quit', '/reset', '/save') else json.dumps({'type': 'text', 'text': line.rstrip('\r\n')})
            incoming.put(line)
        incoming.put(None)

    threading.Thread(target=reader, daemon=True).start()
    remaining, transcript, response, eof = 0, '', '', False
    utterance_samples = 0
    next_idle, idle_count = time.monotonic(), 0
    output_lock = threading.Lock()

    def emit(kind, **values):
        with output_lock:
            display(kind, **values)

    def display(kind, **values):
        if args.action == 'chat':
            if kind == 'text_delta':
                print(values['text'], end='', flush=True)
            elif kind == 'turn_end':
                print(flush=True)
            else:
                print(f'[{kind}] ' + json.dumps(values, ensure_ascii=True), flush=True)
        else:
            print(json.dumps({'type': kind, **values}, ensure_ascii=True), flush=True)

    voice_jobs, cancellations = queue.Queue(maxsize=1), []

    def voice_worker():
        while True:
            job = voice_jobs.get()
            try:
                if job is None:
                    return
                text, cancellation = job
                result = synthesize_speech(model, text, args.voice_max_frames, cancellation)
                if result is not None:
                    samples, report = result
                    path = Path(args.voice_output) / f'voice-{time.time_ns()}.wav'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_wave(path, samples)
                    if not cancellation.is_set():
                        emit('audio_output', path=str(path), sample_rate=16000, **report)
            except Exception as error:
                emit('error', message=f'Speech synthesis: {error}')
            finally:
                voice_jobs.task_done()

    voice_thread = None
    if args.voice_output:
        voice_thread = threading.Thread(target=voice_worker, daemon=True)
        voice_thread.start()

    def cancel_voice():
        for cancellation in cancellations:
            cancellation.set()
        cancellations.clear()

    emit('ready', speech_trained=payload.get('speech_trained', False), device=getattr(model, 'native_backend', str(session.device)))
    try:
        while not eof or remaining:
            try:
                line = incoming.get(timeout=0 if remaining else max(0, next_idle - time.monotonic()))
                if line is None:
                    eof = True
                    continue
                if len(line) > 100000:
                    raise ValueError('Event exceeds 100000 characters')
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError('Event must be a JSON object')
                kind = event.get('type')
                if kind == 'quit':
                    cancel_voice()
                    break
                if kind in ('text', 'audio', 'reset'):
                    cancel_voice()
                    if args.voice_output:
                        emit('audio_cancel')
                if kind in ('text', 'audio', 'reset') and remaining:
                    tail = session.finish()
                    emit('interrupted', text=tail)
                    remaining = 0
                    response = ''
                if kind == 'text':
                    if len(encode(event['text'])) > 8192:
                        raise ValueError('Text event exceeds 8192 UTF-8 bytes')
                    session.hear_text(event['text'])
                    response = ''
                    remaining = args.max_new_tokens
                elif kind == 'audio':
                    if not payload.get('speech_trained'):
                        raise ValueError('Checkpoint has not been trained with paired speech')
                    raw = base64.b64decode(event.get('pcm16_base64', ''), validate=True)
                    if len(raw) > 32000 or len(raw) % 2:
                        raise ValueError('Audio chunk must be at most one second of PCM16 mono at 16000 Hz')
                    utterance_samples += len(raw) // 2
                    if utterance_samples > 30 * 16000:
                        session.reset_audio()
                        utterance_samples, transcript = 0, ''
                        raise ValueError('Audio utterance exceeds 30 seconds; send a final boundary more often')
                    samples = torch.from_numpy(np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768)
                    delta = session.hear_audio(samples, bool(event.get('final', False)))
                    transcript += delta
                    emit('transcript_delta', text=delta)
                    if event.get('final'):
                        emit('transcript', text=transcript)
                        if event.get('respond', True) and transcript:
                            session.hear_text(transcript)
                            response = ''
                            remaining = args.max_new_tokens
                        transcript = ''
                        utterance_samples = 0
                elif kind == 'tick':
                    session.idle(int(event.get('steps', 1)))
                    emit('idle', ticks=event.get('steps', 1))
                elif kind == 'reset':
                    session.reset()
                    transcript = ''
                    utterance_samples = 0
                    emit('reset')
                elif kind == 'save':
                    if not args.save_session:
                        raise ValueError('Set --save-session PATH to enable saving')
                    session.save(args.save_session, checkpoint_id)
                    emit('saved', path=args.save_session)
                elif kind not in ('text', 'audio'):
                    raise ValueError(f'Unknown event type: {kind}')
            except queue.Empty:
                pass
            except (ValueError, KeyError, TypeError) as error:
                emit('error', message=str(error))
            if remaining:
                delta, finished = session.token(args.temperature, args.top_k)
                remaining -= 1
                if delta:
                    response += delta
                    emit('text_delta', text=delta)
                if finished or not remaining:
                    if not finished:
                        tail = session.finish()
                        if tail:
                            response += tail
                            emit('text_delta', text=tail)
                    emit('turn_end', reason='eos' if finished else 'length')
                    if args.voice_output and response.strip():
                        cancellation = threading.Event()
                        cancellations.append(cancellation)
                        try:
                            voice_jobs.put_nowait((response, cancellation))
                        except queue.Full:
                            emit('error', message='Speech worker busy; response text is still available')
                    remaining = 0
            elif not eof and time.monotonic() >= next_idle:
                began = time.monotonic()
                session.idle()
                idle_count += 1
                next_idle = time.monotonic() + 1 / args.idle_hz
                if idle_count % 100 == 0:
                    emit('activity', idle_updates=idle_count, last_update_ms=(time.monotonic() - began) * 1000)
    except KeyboardInterrupt:
        cancel_voice()
        emit('stopped')
    finally:
        if voice_thread:
            voice_jobs.join()
            voice_jobs.put(None)
            voice_thread.join()
        if args.save_session:
            session.save(args.save_session, checkpoint_id)
