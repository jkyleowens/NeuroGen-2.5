"""Causal PCM features, CTC recognition, and a small supervised acoustic decoder.

The deterministic Griffin-Lim vocoder is a baseline, not a neural voice model.
"""
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import encode
from .model import PAD, BOS, EOS

SAMPLE_RATE, FFT, HOP = 16000, 512, 320


def read_wave(path, max_seconds=10):
    with wave.open(str(path), 'rb') as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != SAMPLE_RATE:
            raise ValueError(f'{path}: expected mono 16-bit PCM at 16000 Hz; convert with ffmpeg first')
        if source.getnframes() > max_seconds * SAMPLE_RATE:
            raise ValueError(f'{path}: exceeds audio limit {max_seconds}s')
        raw = source.readframes(source.getnframes())
    if not raw:
        raise ValueError(f'{path}: empty audio')
    return torch.from_numpy(np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768)


def write_wave(path, samples):
    pcm = (samples.detach().cpu().numpy().clip(-1, 1) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as target:
        target.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        target.writeframes(pcm.tobytes())


class Features:
    def __init__(self, bins, device='cpu'):
        self.window = torch.hamming_window(FFT, device=device)
        mel_max = 2595 * np.log10(1 + SAMPLE_RATE / 2 / 700)
        hz = 700 * (10 ** (np.linspace(0, mel_max, bins + 2) / 2595) - 1)
        frequencies = np.linspace(0, SAMPLE_RATE / 2, FFT // 2 + 1)
        filters = [np.maximum(0, np.minimum((frequencies - a) / (b - a), (c - frequencies) / (c - b)))
                   for a, b, c in zip(hz[:-2], hz[1:-1], hz[2:])]
        self.filters = torch.tensor(np.array(filters), dtype=torch.float32, device=device)

    def frames(self, samples):
        spectrum = torch.stft(samples, FFT, HOP, window=self.window, center=False, return_complex=True)
        return torch.log1p(10 * (self.filters @ spectrum.abs()).transpose(0, 1))

    def __call__(self, samples):
        # Pad only the final partial frame. No future-centered analysis windows.
        extra = max(0, FFT - samples.numel())
        extra += (-(samples.numel() + extra - FFT)) % HOP
        return self.frames(F.pad(samples, (0, extra)))

    def synthesize(self, features, iterations=16):
        magnitude = (torch.linalg.pinv(self.filters) @ (features.transpose(0, 1).clamp(0, 12).expm1() / 10)).clamp_min(0)
        spectrum = torch.complex(magnitude, torch.zeros_like(magnitude))
        length = (features.shape[0] - 1) * HOP + FFT
        denominator = magnitude.new_zeros(length)
        for frame in range(features.shape[0]):
            denominator[frame * HOP:frame * HOP + FFT] += self.window.square()
        for _ in range(iterations):
            frames = torch.fft.irfft(spectrum, n=FFT, dim=0).T * self.window
            samples = magnitude.new_zeros(length)
            for frame, values in enumerate(frames):
                samples[frame * HOP:frame * HOP + FFT] += values
            samples /= denominator.clamp_min(1e-6)
            phase = torch.stft(samples, FFT, HOP, window=self.window, center=False, return_complex=True)
            spectrum = magnitude * phase / phase.abs().clamp_min(1e-8)
        return samples.clamp(-1, 1)


class StreamingFeatures:
    def __init__(self, features):
        self.features = features
        self.buffer = features.window.new_empty(0)
        self.emitted = False

    def push(self, samples, final=False):
        if samples.ndim != 1 or not bool(torch.isfinite(samples).all()):
            raise ValueError('Audio samples must be a finite mono vector')
        self.buffer = torch.cat((self.buffer, samples.to(self.buffer)))
        if final and (len(self.buffer) > FFT - HOP or (not self.emitted and len(self.buffer))):
            extra = max(0, FFT - len(self.buffer))
            extra += (-(len(self.buffer) + extra - FFT)) % HOP
            self.buffer = F.pad(self.buffer, (0, extra))
        if len(self.buffer) < FFT:
            return self.buffer.new_empty((0, self.features.filters.shape[0]))
        frames = self.features.frames(self.buffer)
        self.buffer = self.buffer[len(frames) * HOP:]
        self.emitted = True
        return frames


def ctc_logits(logits):
    # Byte symbols plus blank. Dialogue control symbols cannot become transcripts.
    return logits[..., :PAD + 1].log_softmax(-1)


def collapse_ctc(ids, previous=PAD):
    result = []
    for value in ids:
        if value != PAD and value != previous:
            result.append(value)
        previous = value
    return result, previous


def edit_distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        new = [i]
        for j, b in enumerate(hypothesis, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = new
    return row[-1]


class SpeechCorpus:
    def __init__(self, manifest, max_seconds):
        self.rows, self.documents, self.max_seconds = [], set(), max_seconds
        path = Path(manifest)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if not encode(row['text']):
                    raise ValueError('Speech transcript must not be empty')
                wav = (path.parent / row['wav']).resolve()
                samples = read_wave(wav, max_seconds)
                wave_hash = hashlib.sha256(samples.numpy().tobytes()).hexdigest()
                self.documents.add(wave_hash)
                digest.update(wave_hash.encode())
                self.rows.append((wav, row['text']))
        if not self.rows:
            raise ValueError('Empty speech manifest')
        self.sha256 = digest.hexdigest()

    def example(self, index, frontend, device):
        wav, text = self.rows[index % len(self.rows)]
        features = frontend(read_wave(wav, self.max_seconds).to(device)).unsqueeze(0)
        return features, text


def speech_objectives(model, features, text):
    ids = encode(text)
    minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
    if features.shape[1] < minimum:
        raise ValueError(f'Audio has {features.shape[1]} frames but CTC needs at least {minimum} for this transcript')
    target = torch.tensor(ids, dtype=torch.long, device=features.device)
    logits, _, activity = model.listen(features)
    loss_ctc = F.ctc_loss(ctc_logits(logits).transpose(0, 1), target,
                          (features.shape[1],), (len(ids),), blank=PAD, zero_infinity=False)
    tokens = torch.tensor([[BOS] + ids + [EOS]], device=features.device)
    _, _, _, hidden = model(tokens)
    shifted = F.pad(features[:, :-1], (0, 0, 1, 0))
    predicted, stop, _ = model.acoustic_decode(shifted, hidden)
    stop_target = torch.zeros_like(stop)
    stop_target[:, -1] = 1
    loss_mel = F.l1_loss(predicted, features)
    loss_stop = F.binary_cross_entropy_with_logits(stop, stop_target,
                                                  pos_weight=stop.new_tensor(min(20, features.shape[1] - 1)))
    decoded, _ = collapse_ctc(logits[0, :, :PAD + 1].argmax(-1).tolist())
    transcript = bytes(decoded).decode('utf-8', errors='replace')
    return loss_ctc, loss_mel, loss_stop, activity, transcript
