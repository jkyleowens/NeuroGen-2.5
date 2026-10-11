"""Bounded-memory local corpus ingestion; no network downloads or token vocabulary fit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import PAD, BOS, EOS, USER, ASSISTANT, SYSTEM

ROLES = {'user': USER, 'assistant': ASSISTANT, 'system': SYSTEM}


def encode(text):
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    return list(text.encode('utf-8'))


def record_tokens(record):
    tokens, mask = [BOS], [False]
    if 'messages' in record:
        for message in record['messages']:
            if message.get('role') not in ROLES:
                raise ValueError('Message role must be user, assistant, or system')
            content = encode(message['content']) + [EOS]
            tokens += [ROLES[message['role']]] + content
            mask += [False] + [message['role'] == 'assistant'] * len(content)
    else:
        content = encode(record['text']) + [EOS]
        tokens += content
        mask += [True] * len(content)
    if not any(mask):
        raise ValueError('A record needs text or an assistant response to supervise')
    return tokens, mask


class Corpus:
    def __init__(self, source, cache):
        source, cache = Path(source), Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        self.index, self.documents = [], set()
        digest, offset = hashlib.sha256(), 0
        with source.open('rb') as src, (cache / 'tokens.bin').open('wb') as tok, (cache / 'mask.bin').open('wb') as mask:
            for line in src:
                digest.update(line)
                if not line.strip():
                    continue
                record = json.loads(line) if source.suffix == '.jsonl' else {'text': line.decode('utf-8').rstrip('\r\n')}
                ids, supervised = record_tokens(record)
                array = np.asarray(ids, dtype='<u2')
                array.tofile(tok)
                np.asarray(supervised, dtype=np.uint8).tofile(mask)
                self.index.append((offset, len(ids)))
                self.documents.add(hashlib.sha256(array.tobytes()).hexdigest())
                offset += len(ids)
        if not self.index:
            raise ValueError(f'Empty corpus: {source}')
        self.sha256 = digest.hexdigest()
        self.tokens = np.memmap(cache / 'tokens.bin', mode='r', dtype='<u2')
        self.mask = np.memmap(cache / 'mask.bin', mode='r', dtype=np.uint8)

    def batch(self, batch_size, length, generator, device):
        x = torch.full((batch_size, length), PAD, dtype=torch.long)
        y = torch.full_like(x, -100)
        for row in range(batch_size):
            for _ in range(1000):
                index = int(torch.randint(len(self.index), (), generator=generator))
                offset, size = self.index[index]
                start = int(torch.randint(max(1, size - length), (), generator=generator))
                end = min(size, start + length + 1)
                ids = np.array(self.tokens[offset + start:offset + end], dtype=np.int64)
                supervised = np.array(self.mask[offset + start + 1:offset + end], dtype=bool)
                if supervised.any():
                    count = len(ids) - 1
                    x[row, :count] = torch.from_numpy(ids[:-1])
                    labels = torch.from_numpy(ids[1:].copy())
                    labels[~torch.from_numpy(supervised)] = -100
                    y[row, :count] = labels
                    break
            else:
                raise ValueError('Could not sample supervised targets; use longer sequences or shorter prompts')
        return x.to(device), y.to(device)


def disjoint(train, validation):
    if train.documents & validation.documents:
        raise ValueError('Training and validation contain identical documents; use a held-out split')
