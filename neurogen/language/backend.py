"""CPU/CUDA/ROCm selection with an executable GPU diagnostic."""
import json
import platform
import sys

import torch

from .model import LanguageConfig, LanguageModel


def device_for(name):
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name not in ('cuda', 'rocm'):
        raise ValueError('Device must be auto, cpu, cuda, or rocm')
    if not torch.version.cuda and not torch.version.hip:
        raise ValueError(
            f'{name} requested, but PyTorch {torch.__version__} is a CPU-only build in {sys.executable}. '
            'Installing the CUDA toolkit alone does not enable GPU support in this build. '
            'Replace torch in this Python environment with the appropriate NVIDIA CUDA or AMD ROCm build; '
            'see LANGUAGE.md, then restart the GUI and reload the model.')
    if name == 'rocm' and not torch.version.hip:
        raise ValueError('ROCm requested but this is not a HIP/ROCm PyTorch build')
    if name == 'cuda' and torch.version.hip:
        raise ValueError('CUDA requested but this is a HIP/ROCm build; select rocm or auto')
    if not torch.cuda.is_available():
        try:
            torch.cuda.init()
        except (RuntimeError, AssertionError, OSError) as error:
            detail = str(error)
        else:
            detail = 'No usable GPU was detected.'
        raise ValueError(
            f'{name} requested, but PyTorch {torch.__version__} cannot access the GPU in {sys.executable}. '
            f'{detail} Check the driver, GPU visibility and process permissions; '
            'run language doctor in the same environment as the GUI. See LANGUAGE.md.')
    return torch.device('cuda')  # HIP intentionally uses the torch.cuda API.


def environment(device):
    result = {'python': platform.python_version(), 'python_executable': sys.executable,
              'torch': str(torch.__version__), 'torch_path': torch.__file__,
              'device': str(device), 'cuda_build': torch.version.cuda, 'hip_build': torch.version.hip,
              'gpu_available': torch.cuda.is_available(),
              'platform': platform.platform(), 'precision': 'float32'}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        architecture = (getattr(props, 'gcnArchName', 'unknown') if torch.version.hip else
                        'sm_' + ''.join(map(str, torch.cuda.get_device_capability(device))))
        result.update(gpu=props.name, architecture=architecture,
                      free_mib=free / 1024**2, total_mib=total / 1024**2)
    return result


def doctor(args):
    torch.set_num_threads(args.threads)
    try:
        device = device_for(args.device)
    except ValueError as error:
        report = environment(torch.device('cpu'))
        report.update(requested_device=args.device, diagnostic_passed=False, gpu_tested=False,
                      error=str(error))
        print(json.dumps(report, indent=2))
        raise
    report = environment(device)
    report.update(LanguageModel(LanguageConfig.preset(args.preset)).specification())
    report.update(requested_device=args.device, diagnostic_model_preset='tiny')
    torch.manual_seed(7)
    cpu = LanguageModel(LanguageConfig.preset('tiny'))
    x = torch.tensor([[257, 104, 101, 108, 108, 111]])
    expected = cpu(x)[0]
    expected.square().mean().backward()
    target = LanguageModel(cpu.config).to(device)
    target.load_state_dict(cpu.state_dict())
    actual = target(x.to(device))[0]
    actual.square().mean().backward()
    error = float((expected.detach() - actual.detach().cpu()).abs().max())
    from .audio import speech_objectives
    features = torch.linspace(0, 2, 24 * cpu.config.mel_bins).reshape(1, 24, -1)
    expected_speech = speech_objectives(cpu, features, 'hi')
    actual_speech = speech_objectives(target, features.to(device), 'hi')
    speech_loss = actual_speech[0] * .1 + actual_speech[1] + actual_speech[2]
    speech_loss.backward()
    speech_error = max(abs(float(a.detach()) - float(b.detach())) for a, b in zip(expected_speech[:3], actual_speech[:3]))
    finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in target.parameters())
    report.update(forward_max_error=error, speech_loss_max_error=speech_error, backward_finite=finite,
                  diagnostic_passed=error < .01 and speech_error < .1 and finite,
                  gpu_tested=device.type == 'cuda',
                  deployment_note='Run this on the BC-250 itself. A CPU pass does not validate ROCm/gfx1013.')
    print(json.dumps(report, indent=2))
    if not report['diagnostic_passed']:
        raise ValueError('Device forward/backward diagnostic failed')
