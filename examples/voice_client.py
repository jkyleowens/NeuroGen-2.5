"""Optional local microphone/playback client for NeuroGen's JSONL protocol.

Uses a simple energy gate, not learned VAD or echo cancellation. Use headphones.
No recordings are sent to a remote service. Generated WAVs stay in --output.
"""
import argparse
import base64
from collections import deque
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import wave

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'rocm'), default='auto')
    parser.add_argument('--output', default='runs/voice-session')
    parser.add_argument('--input-device', type=int)
    parser.add_argument('--output-device', type=int)
    parser.add_argument('--threshold', type=float, default=.015, help='Microphone RMS energy gate')
    parser.add_argument('--max-turn-seconds', type=float, default=5.)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or not 1 <= args.max_turn_seconds <= 20:
        parser.error('threshold must be 0..1 and max-turn-seconds 1..20')
    try:
        import sounddevice as sd
    except ImportError:
        parser.error("Install the optional voice dependencies: pip install -e '.[voice]'")
    output = Path(args.output).resolve()
    process = subprocess.Popen([sys.executable, '-m', 'neurogen', 'language', 'stream',
        '--checkpoint', args.checkpoint, '--device', args.device, '--voice-output', str(output),
        '--save-session', str(output.parent / (output.name + '-state.pt'))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    ready, failed, overflow = threading.Event(), threading.Event(), threading.Event()
    incoming = queue.Queue(maxsize=10)

    def outputs():
        for line in process.stdout:
            event = json.loads(line)
            kind = event['type']
            if kind == 'ready':
                ready.set()
            elif kind == 'audio_cancel':
                sd.stop()
            elif kind == 'audio_output':
                with wave.open(event['path'], 'rb') as source:
                    audio = np.frombuffer(source.readframes(source.getnframes()), dtype='<i2').copy()
                sd.play(audio, samplerate=16000, device=args.output_device, blocking=False)
            elif kind == 'text_delta':
                print(event['text'], end='', flush=True)
            elif kind == 'turn_end':
                print(flush=True)
            elif kind == 'transcript':
                print('\nHeard: ' + event['text'], flush=True)
            elif kind == 'error':
                print(event['message'], file=sys.stderr, flush=True)
        failed.set()
        ready.set()

    def callback(data, frames, timing, status):
        if status:
            overflow.set()
        try:
            incoming.put_nowait(bytes(data))
        except queue.Full:
            overflow.set()

    def send(raw, final=False):
        process.stdin.write(json.dumps({'type': 'audio', 'pcm16_base64': base64.b64encode(raw).decode(), 'final': final}) + '\n')
        process.stdin.flush()

    threading.Thread(target=outputs, daemon=True).start()
    try:
        if not ready.wait(60) or failed.is_set():
            raise RuntimeError('Speech runtime failed to start; check its checkpoint and device diagnostic')
        print('Microphone active. Speak, then pause for 0.6 s. Ctrl-C exits. Use headphones.', flush=True)
        active, silence, samples = False, 0, 0
        preroll = deque(maxlen=2)
        with sd.RawInputStream(samplerate=16000, blocksize=3200, channels=1,
                               dtype='int16', device=args.input_device, callback=callback):
            while not failed.is_set():
                raw = incoming.get(timeout=2)
                if overflow.is_set():
                    raise RuntimeError('Audio input overflow or runtime too slow; stopping to avoid silently dropping audio')
                values = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
                voiced = float(np.sqrt(np.mean(values * values))) >= args.threshold
                if not active:
                    preroll.append(raw)
                    if not voiced:
                        continue
                    active, silence, samples = True, 0, 0
                    sd.stop()
                    for block in preroll:
                        send(block)
                        samples += len(block) // 2
                    preroll.clear()
                else:
                    send(raw)
                    samples += len(raw) // 2
                silence = 0 if voiced else silence + 1
                if silence >= 3 or samples >= args.max_turn_seconds * 16000:
                    send(b'', final=True)
                    active = False
    except KeyboardInterrupt:
        pass
    finally:
        sd.stop()
        try:
            process.stdin.write('{"type":"quit"}\n')
            process.stdin.flush()
            process.stdin.close()
            process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()
            process.wait(timeout=10)


if __name__ == '__main__':
    main()
