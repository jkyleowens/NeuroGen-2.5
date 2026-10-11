"""Hybrid laminar spiking recurrent language core, portable PyTorch operations.

This is a trainable abstraction, not a port of the physical conductance simulator.
Graded membrane readout and surrogate gradients provide supervised credit assignment.
"""
from dataclasses import asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from neurogen.anatomy import LAYERS
from .config import LanguageConfig

PAD, BOS, EOS, USER, ASSISTANT, SYSTEM, AUDIO, IDLE = range(256, 264)
VOCAB = 264


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + 5 * x.abs()).square()


def detached(state):
    return tuple(tuple(x.detach() for x in region) for region in state)


class ColumnRegion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layer, excitation = zip(*[(layer, exc) for layer, ne, ni in LAYERS for exc in [True] * ne + [False] * ni])
        layer, excitation = torch.tensor(layer), torch.tensor(excitation)
        preferred = ((4, 2), (4, 3), (2, 3), (3, 5), (5, 6), (6, 4))
        mask = layer[:, None] == layer[None, :]
        for pre, post in preferred:
            mask |= (layer[:, None] == post) & (layer[None, :] == pre)
        # Short-range inhibitory axons, no autapses. Rows=postsynaptic.
        mask |= (~excitation[None, :]) & ((layer[:, None] - layer[None, :]).abs() <= 1)
        mask.fill_diagonal_(False)
        self.register_buffer('mask', mask.float())
        self.register_buffer('sign', torch.where(excitation, 1., -1.))
        # Normalized units: faster, more excitable, non-adapting inhibitory cells.
        self.register_buffer('membrane_decay', torch.where(excitation, .6, .4))
        self.register_buffer('threshold', torch.where(excitation, 1., .6))
        self.register_buffer('baseline', torch.where(excitation, .15, .35))
        self.register_buffer('adaptation_increment', torch.where(excitation, .03, 0.))
        self.register_buffer('input_ids', torch.where((layer == 4) & excitation)[0])
        self.register_buffer('feedback_ids', torch.where((layer >= 5) | (layer == 1))[0])
        self.register_buffer('output_ids', torch.where(excitation)[0])
        self.register_buffer('layer', layer)
        c, w = config.columns, config.width
        self.raw_weight = nn.Parameter(torch.full((c, 64, 64), -1.8))
        self.input = nn.Linear(w, c * len(self.input_ids))
        self.feedback = nn.Linear(w, c * len(self.feedback_ids), bias=False)
        self.output = nn.Linear(c * len(self.output_ids) * 2, w)
        self.norm = nn.LayerNorm(w)
        self.memory_decay = nn.Parameter(torch.full((w,), 2.9))

    def initial(self, batch, like):
        shape = (batch, self.config.columns, 64)
        return tuple(like.new_zeros(shape) for _ in range(4)) + (like.new_zeros((batch, self.config.width)), like.new_zeros(shape))

    def forward(self, x, state, sensory=True):
        voltage, trace, adaptation, last_spikes, memory, counts = state
        c = self.config.columns
        current = self.baseline.expand_as(voltage).clone()
        if sensory:
            current[..., self.input_ids] = 3 * torch.sigmoid(self.input(x).reshape(-1, c, len(self.input_ids)))
        current[..., self.feedback_ids] += .5 * torch.tanh(self.feedback(memory).reshape(-1, c, len(self.feedback_ids)))
        weights = F.softplus(self.raw_weight) * self.mask * self.sign
        total_spikes = voltage.new_zeros(())
        for _ in range(self.config.ticks):
            recurrent = torch.einsum('bci,coi->bco', trace, weights)
            # Horizontal communication between adjacent columns, no wrap-around.
            neighbor = torch.zeros_like(trace)
            neighbor[:, 1:] += trace[:, :-1] * (self.sign > 0)
            neighbor[:, :-1] += trace[:, 1:] * (self.sign > 0)
            voltage = self.membrane_decay * voltage + (1 - self.membrane_decay) * (current + recurrent + .08 * neighbor)
            spikes = Spike.apply(voltage - self.threshold - adaptation) * (1 - last_spikes.detach())
            voltage = voltage * (1 - spikes.detach())
            adaptation = .98 * adaptation + self.adaptation_increment * spikes
            trace = .8 * trace + .2 * spikes
            total_spikes = total_spikes + spikes.mean()
            last_spikes = spikes
            counts = counts + spikes.detach()
        # Read graded voltage as well as spike traces, explicitly a hybrid model.
        features = torch.cat((voltage[..., self.output_ids].flatten(1), trace[..., self.output_ids].flatten(1)), -1)
        h = self.norm(self.output(features) + .25 * memory)
        decay = .5 + .499 * torch.sigmoid(self.memory_decay)
        memory = decay * memory + (1 - decay) * h
        return h, (voltage, trace, adaptation, last_spikes, memory, counts), total_spikes / self.config.ticks


class LanguageModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LanguageConfig()
        c = self.config
        self.embedding = nn.Embedding(VOCAB, c.width, padding_idx=PAD)
        self.regions = nn.ModuleList(ColumnRegion(c) for _ in range(c.regions))
        self.head = nn.Linear(c.width, VOCAB)
        self.audio_input = nn.Linear(c.mel_bins, c.width)
        # A separate conventional acoustic decoder; it is not labeled biological.
        self.speech_decoder = nn.GRU(c.mel_bins + c.width, c.speech_width, batch_first=True)
        self.speech_query = nn.Linear(c.speech_width, c.width)
        self.speech_output = nn.Linear(c.speech_width + c.width, c.mel_bins)
        self.speech_stop = nn.Linear(c.speech_width + c.width, 1)

    def initial(self, batch):
        return tuple(region.initial(batch, self.embedding.weight) for region in self.regions)

    def advance(self, inputs, state=None, idle=False):
        if state is None:
            state = self.initial(len(inputs))
        h, states, activity = inputs, [], []
        for index, (region, previous) in enumerate(zip(self.regions, state)):
            # During silence only external input is removed; regional communication continues.
            h, new_state, spikes = region(h, previous, sensory=not idle or index > 0)
            states.append(new_state)
            activity.append(spikes)
        return h, tuple(states), torch.stack(activity).mean()

    def step(self, tokens, state=None):
        h, state, activity = self.advance(self.embedding(tokens), state)
        for _ in range(self.config.idle_steps):
            h, state, other = self.advance(torch.zeros_like(h), state, idle=True)
            activity = activity + other
        return self.head(h), state, activity / (1 + self.config.idle_steps), h

    def forward(self, tokens, state=None):
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError('tokens must be int64 [batch, time]')
        outputs, hidden, activity = [], [], []
        for token in tokens.unbind(1):
            logits, state, spikes, h = self.step(token, state)
            outputs.append(logits)
            hidden.append(h)
            activity.append(spikes)
        if not outputs:
            raise ValueError('Need at least one input token')
        return torch.stack(outputs, 1), state, torch.stack(activity).mean(), torch.stack(hidden, 1)

    def listen(self, features, state=None):
        logits, activity = [], []
        for frame in features.unbind(1):
            h, state, spikes = self.advance(self.audio_input(frame), state)
            logits.append(self.head(h))
            activity.append(spikes)
        return torch.stack(logits, 1), state, torch.stack(activity).mean()

    def acoustic_decode(self, shifted_mels, text_hidden, hidden=None):
        context = text_hidden.mean(1, keepdim=True).expand(-1, shifted_mels.shape[1], -1)
        decoded, hidden = self.speech_decoder(torch.cat((shifted_mels, context), -1), hidden)
        scores = self.speech_query(decoded) @ text_hidden.transpose(1, 2) / math.sqrt(self.config.width)
        attended = scores.softmax(-1) @ text_hidden
        combined = torch.cat((decoded, attended), -1)
        return F.softplus(self.speech_output(combined)), self.speech_stop(combined).squeeze(-1), hidden

    def specification(self):
        parameters = sum(p.numel() for p in self.parameters())
        return {'config': asdict(self.config), 'parameters': parameters,
                'spiking_neurons': self.config.regions * self.config.columns * 64,
                'fp32_adam_parameter_state_mib': parameters * 16 / 1024**2,
                'note': 'Parameter-state estimate excludes activations, runtime workspaces and audio buffers.'}

    def activity_report(self, state, internal_ticks):
        result = {}
        for index, (region, previous) in enumerate(zip(self.regions, state)):
            counts = previous[5]
            for population, mask in (('excitatory', region.sign > 0), ('inhibitory', region.sign < 0)):
                result[f'region_{index}_{population}'] = float(counts[..., mask].mean() / internal_ticks)
        return result
