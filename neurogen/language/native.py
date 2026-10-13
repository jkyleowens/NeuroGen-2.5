"""C++ inference adapter; Python owns checkpoint I/O and the acoustic decoder.

State tensors deliberately use the reference model's layout. Native handles own
immutable weight snapshots, not session state, so sessions and speech workers do
not share recurrent memory. The C ABI serializes use of each GPU command queue.
"""
import ctypes as ct
import os
from pathlib import Path
import weakref

import numpy as np
import torch

from .model import LanguageModel, ColumnRegion, VOCAB
from .config import LanguageConfig

NATIVE_DEVICES = ('native-cpu', 'native-cuda', 'vulkan')
F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')


def library():
    explicit = os.environ.get('NEUROGEN_NATIVE_LIBRARY')
    root = Path(__file__).resolve().parents[2]
    candidates = [Path(explicit)] if explicit else [root / 'build/native' / name for name in
                    ('libneurogen_native.so', 'libneurogen_native.dylib', 'neurogen_native.dll')]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ValueError('Native engine not built. See NATIVE.md; set NEUROGEN_NATIVE_LIBRARY for a custom build path.')
    lib = ct.CDLL(str(path.resolve()))
    lib.ng_create.argtypes = [ct.c_int] * 5 + [F32, ct.c_size_t, ct.c_char_p]
    lib.ng_create.restype = ct.c_void_p
    lib.ng_destroy.argtypes = [ct.c_void_p]
    lib.ng_destroy.restype = None
    lib.ng_error.argtypes = []
    lib.ng_error.restype = ct.c_char_p
    lib.ng_backend.argtypes = [ct.c_void_p]
    lib.ng_backend.restype = ct.c_char_p
    lib.ng_step.argtypes = [ct.c_void_p, ct.c_int, F32, ct.c_size_t, F32, F32, F32, F32]
    lib.ng_step.restype = ct.c_int
    lib.ng_advance.argtypes = [ct.c_void_p, F32, F32, ct.c_size_t, ct.c_int, F32, F32, F32]
    lib.ng_advance.restype = ct.c_int
    return lib


def packed_weights(model):
    tensors = [model.embedding.weight]
    for r in model.regions:
        tensors.extend((r.raw_weight, r.input.weight, r.input.bias, r.feedback.weight,
                        r.output.weight, r.output.bias, r.norm.weight, r.norm.bias, r.memory_decay))
    tensors.extend((model.head.weight, model.head.bias))
    return np.concatenate([t.detach().cpu().float().numpy().reshape(-1) for t in tensors])


class NativeLanguageModel(LanguageModel):
    @classmethod
    def from_model(cls, model, backend='native-cpu'):
        if backend not in NATIVE_DEVICES:
            raise ValueError(f'Unknown native device: {backend}')
        if isinstance(model, cls):
            raise ValueError('Model already has a native engine')
        # Retain parameter names and speech modules for existing checkpoints/UI.
        model.eval().cpu().float()
        model.__class__ = cls
        model._native_lib = library()
        model._native_device = backend
        model._rebuild_native()
        model.requires_grad_(False)
        return model

    def _rebuild_native(self):
        c = self.config
        # These buffers define the architecture, not trainable weights. Reject
        # custom buffer edits rather than silently running different equations.
        with torch.random.fork_rng(devices=[]):
            canonical = ColumnRegion(LanguageConfig(columns=1, width=1))
        for region in self.regions:
            for name, expected in canonical.named_buffers():
                if not torch.equal(getattr(region, name).cpu(), expected):
                    raise ValueError(f'Native architecture buffer differs from simplified model: {name}')
        weights = packed_weights(self)
        backend = {'native-cpu': 'cpu', 'native-cuda': 'cuda', 'vulkan': 'vulkan'}[self._native_device]
        handle = self._native_lib.ng_create(c.columns, c.width, c.regions, c.ticks, c.idle_steps,
                                             weights, len(weights), backend.encode())
        if not handle:
            raise ValueError(self._native_lib.ng_error().decode())
        if hasattr(self, '_native_finalizer'):
            self._native_finalizer()
        self._native_handle = handle
        self._native_finalizer = weakref.finalize(self, self._native_lib.ng_destroy, handle)
        self.native_backend = self._native_lib.ng_backend(handle).decode()
        self._native_versions = tuple(p._version for p in list(self.parameters()) + list(self.buffers()))

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        if hasattr(self, '_native_lib'):
            self._rebuild_native()
        return result

    def train(self, mode=True):
        if mode and hasattr(self, '_native_handle'):
            raise ValueError('Native execution is inference-only. Train with PyTorch cpu/cuda/rocm, then reload the checkpoint.')
        return super().train(mode)

    def _call(self, inputs, state, idle, tokens):
        if self.embedding.weight.device.type != 'cpu' or self.embedding.weight.dtype != torch.float32:
            raise ValueError('Native adapter tensors must remain CPU float32; the C++ backend manages GPU memory')
        if tuple(p._version for p in list(self.parameters()) + list(self.buffers())) != self._native_versions:
            raise ValueError('Native weights changed; reload using load_state_dict before inference')
        batch = len(inputs)
        if batch < 1:
            raise ValueError('Need at least one batch item')
        if state is None:
            state = self.initial(batch)
        c = self.config
        shapes = [(batch, c.columns, 64)] * 4 + [(batch, c.width), (batch, c.columns, 64)]
        if len(state) != c.regions or any(len(r) != 6 for r in state):
            raise ValueError('Invalid native region state')
        for region in state:
            for tensor, shape in zip(region, shapes):
                if tuple(tensor.shape) != shape or tensor.device.type != 'cpu' or tensor.dtype != torch.float32:
                    raise ValueError('Invalid native state shape/device/dtype')
        packed = np.ascontiguousarray(torch.cat([t.detach().reshape(batch, -1) for r in state for t in r], 1).numpy())
        next_state = np.empty_like(packed)
        hidden = np.empty((batch, c.width), dtype=np.float32)
        logits = np.empty((batch, VOCAB), dtype=np.float32)
        activity = np.empty(batch, dtype=np.float32)
        for b in range(batch):
            if tokens:
                code = self._native_lib.ng_step(self._native_handle, int(inputs[b]), packed[b], packed.shape[1],
                                                next_state[b], logits[b], hidden[b], activity[b:b+1])
            else:
                x = np.ascontiguousarray(inputs[b].detach().cpu().float().numpy())
                code = self._native_lib.ng_advance(self._native_handle, x, packed[b], packed.shape[1], int(idle),
                                                   next_state[b], hidden[b], activity[b:b+1])
            if code:
                raise ValueError(self._native_lib.ng_error().decode())
        result, offset = [], 0
        # Each returned tensor is contiguous and independent of the caller's state.
        for _ in range(c.regions):
            fields = []
            for shape in shapes:
                size = int(np.prod(shape[1:]))
                fields.append(torch.from_numpy(next_state[:, offset:offset+size].copy()).reshape(shape))
                offset += size
            result.append(tuple(fields))
        return torch.from_numpy(logits), tuple(result), torch.from_numpy(activity).mean(), torch.from_numpy(hidden)

    def step(self, tokens, state=None):
        if tokens.ndim != 1 or tokens.dtype != torch.long or bool(((tokens < 0) | (tokens >= VOCAB)).any()):
            raise ValueError('Native tokens must be int64 [batch] in 0..263')
        return self._call(tokens, state, False, True)

    def advance(self, inputs, state=None, idle=False):
        if inputs.ndim != 2 or inputs.shape[1] != self.config.width:
            raise ValueError('Native inputs must have shape [batch, width]')
        _, state, activity, hidden = self._call(inputs, state, idle, False)
        return hidden, state, activity


def inference_model(model, device):
    return NativeLanguageModel.from_model(model, device) if device in NATIVE_DEVICES else model
