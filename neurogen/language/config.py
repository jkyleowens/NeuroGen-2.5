"""Language architecture configuration; safe to import without PyTorch."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageConfig:
    columns: int = 32
    width: int = 768
    regions: int = 4
    ticks: int = 2
    idle_steps: int = 1
    mel_bins: int = 64
    speech_width: int = 256

    def __post_init__(self):
        for name in ('columns', 'width', 'regions', 'ticks', 'mel_bins', 'speech_width'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.idle_steps) is not int or not 0 <= self.idle_steps <= 16:
            raise ValueError('idle_steps must be between 0 and 16')

    @classmethod
    def preset(cls, name):
        presets = {'tiny': dict(columns=1, width=32, regions=2, speech_width=32, idle_steps=0),
                   'laptop': dict(columns=4, width=128, regions=3, speech_width=128),
                   'bc250': {}, 'large': dict(columns=64, width=1024, regions=4)}
        if name not in presets:
            raise ValueError(f'Unknown preset: {name}')
        return cls(**presets[name])

