"""Executable reference comparison; reports software Vulkan without claiming a GPU test."""
import copy
import json
import torch
from .config import LanguageConfig
from .model import LanguageModel
from .native import NativeLanguageModel


def doctor(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    reference = LanguageModel(LanguageConfig(columns=2, width=32, regions=2, ticks=3, idle_steps=1)).eval()
    target = NativeLanguageModel.from_model(copy.deepcopy(reference), args.device)
    a = b = None
    error = 0.
    finite = True
    with torch.no_grad():
        for token in [257, 104, 101, 108, 108, 111] * 3:
            x = torch.tensor([token])
            la, a, aa, ha = reference.step(x, a)
            lb, b, ab, hb = target.step(x, b)
            pairs = [(la, lb), (ha, hb), (aa, ab)] + [(x, y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
            finite = finite and all(bool(torch.isfinite(y).all()) for _, y in pairs)
            error = max(error, *(float((x-y).abs().max()) for x, y in pairs))
        for _ in range(10):
            ha, a, aa = reference.advance(torch.zeros(1, 32), a, idle=True)
            hb, b, ab = target.advance(torch.zeros(1, 32), b, idle=True)
            error = max(error, float((ha-hb).abs().max()),
                        *(float((x-y).abs().max()) for ra, rb in zip(a, b) for x, y in zip(ra, rb)))
    finite = finite and all(bool(torch.isfinite(t).all()) for region in b for t in region)
    passed = finite and error < 2e-4
    print(json.dumps({'backend': target.native_backend, 'diagnostic_model': '2 columns, 2 regions, width 32',
                      'forward_and_state_max_error': error, 'diagnostic_passed': passed,
                      'finite': finite, 'training_supported': False, 'software_vulkan': '(software)' in target.native_backend}, indent=2))
    if not passed:
        raise ValueError('Native parity diagnostic failed')
