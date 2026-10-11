"""Configuration validation and architecture estimates without GPU allocations."""
from dataclasses import asdict
import json
import math
from pathlib import Path

from neurogen.language.config import LanguageConfig


def parse_config(text, biological=False):
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('Configuration must be a JSON object')
    try:
        if biological:
            from neurogen.biology import BiologyConfig
            return asdict(BiologyConfig(**data))
        return asdict(LanguageConfig(**data))
    except TypeError as error:
        raise ValueError(str(error)) from error


def estimate(config):
    c = LanguageConfig(**config)
    columns, width, regions, mel, speech = c.columns, c.width, c.regions, c.mel_bins, c.speech_width
    # Matches the laminar core and shared text/audio heads, including biases.
    regional = 4096 * columns + 141 * columns * width + 12 * columns + 4 * width
    parameters = (regions * regional + 528 * width + 264 + (mel + 1) * width
                  + 3 * speech * (mel + width + speech) + 6 * speech
                  + speech * width + width + (speech + width) * mel + mel + speech + width + 1)
    return {'parameters': parameters, 'neurons': regions * columns * 64,
            'weights_mib': parameters * 4 / 1024**2, 'adam_mib': parameters * 16 / 1024**2}


def inference_settings(values):
    result = dict(temperature=float(values.get('temperature', .7)), top_k=int(values.get('top_k', 40)),
                  max_new_tokens=int(values.get('max_new_tokens', 256)), idle_hz=float(values.get('idle_hz', 10)),
                  continuous=bool(values.get('continuous', False)))
    if not math.isfinite(result['temperature']) or not 0 <= result['temperature'] <= 10:
        raise ValueError('Temperature must be between 0 and 10; 0 selects greedy decoding')
    if not 1 <= result['top_k'] <= 257:
        raise ValueError('Top-k must be between 1 and 257')
    if not 1 <= result['max_new_tokens'] <= 8192:
        raise ValueError('Output limit must be between 1 and 8192 bytes')
    if not math.isfinite(result['idle_hz']) or not 0 < result['idle_hz'] <= 100:
        raise ValueError('Idle update rate must be greater than 0 and at most 100')
    return result


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
