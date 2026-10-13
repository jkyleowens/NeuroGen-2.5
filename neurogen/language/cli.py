"""Keep torch optional: parser construction imports no tensor libraries."""


def add_language_parser(commands):
    language = commands.add_parser('language', help='Train and run the hybrid spiking language/speech model')
    actions = language.add_subparsers(dest='action', required=True)

    def common(parser, native=False):
        parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'rocm') + (('native-cpu', 'native-cuda', 'vulkan') if native else ()), default='auto')
        parser.add_argument('--threads', type=int, default=1)

    doctor = actions.add_parser('doctor', help='Inspect hardware and execute a forward/backward device check')
    common(doctor, native=True)
    doctor.add_argument('--preset', choices=('tiny', 'laptop', 'bc250', 'large'), default='bc250')

    train = actions.add_parser('train', help='Supervised text/dialogue and optional paired-speech training')
    common(train)
    train.add_argument('--data-source', choices=('slimpajama', 'files'), help='Default: automatically prepare SlimPajama; local paths imply files')
    train.add_argument('--train', help='Optional local UTF-8 text or JSONL; requires --validation')
    train.add_argument('--validation', help='Optional local held-out corpus; requires --train')
    from .corpus import DEFAULT_DATASET
    train.add_argument('--dataset-id', default=DEFAULT_DATASET, help='SlimPajama repository on Hugging Face')
    train.add_argument('--dataset-revision', default='main')
    train.add_argument('--dataset-seed', type=int, help='Sampling seed; omitted = a fresh random subset each run')
    train.add_argument('--train-documents', type=int, default=2000)
    train.add_argument('--validation-documents', type=int, default=128)
    train.add_argument('--max-document-bytes', type=int, default=8192)
    train.add_argument('--shuffle-buffer', type=int, default=128)
    train.add_argument('--speech-train', help='JSONL wav/text manifest')
    train.add_argument('--speech-validation', help='Held-out JSONL wav/text manifest')
    train.add_argument('--max-audio-seconds', type=int, default=6)
    train.add_argument('--preset', choices=('tiny', 'laptop', 'bc250', 'large'), default='bc250')
    train.add_argument('--config', help='JSON overrides for the selected size preset')
    train.add_argument('--steps', type=int, default=1000)
    train.add_argument('--batch-size', type=int, default=2)
    train.add_argument('--sequence-length', type=int, default=96)
    train.add_argument('--accumulate', type=int, default=4)
    train.add_argument('--learning-rate', type=float, default=.001)
    train.add_argument('--ctc-weight', type=float, default=.1)
    train.add_argument('--mel-weight', type=float, default=1.)
    train.add_argument('--stop-weight', type=float, default=.1)
    train.add_argument('--eval-every', type=int, default=25)
    train.add_argument('--eval-batches', type=int, default=4)
    train.add_argument('--memory-budget-mib', type=float, default=6144)
    train.add_argument('--seed', type=int, default=1)
    train.add_argument('--output')
    source = train.add_mutually_exclusive_group()
    source.add_argument('--resume', help='Resume checkpoint into a new output directory')
    source.add_argument('--initialize-from', help='Start a new optimizer/corpus run from checkpoint weights')

    for action in ('stream', 'chat', 'listen', 'speak'):
        parser = actions.add_parser(action)
        common(parser, native=True)
        parser.add_argument('--checkpoint', required=True)
        if action in ('stream', 'chat'):
            parser.add_argument('--max-new-tokens', type=int, default=256, help='Maximum generated UTF-8 bytes per turn')
            parser.add_argument('--temperature', type=float, default=.7)
            parser.add_argument('--top-k', type=int, default=40)
            parser.add_argument('--idle-hz', type=float, default=10, help='Wall-clock idle updates/second, not neuron firing Hz')
            parser.add_argument('--seed', type=int, default=1)
            parser.add_argument('--save-session')
            parser.add_argument('--resume-session')
            parser.add_argument('--voice-output', help='Synthesize completed responses asynchronously into this WAV directory')
            parser.add_argument('--voice-max-frames', type=int, default=300)
        elif action == 'listen':
            parser.add_argument('--wav', required=True)
            parser.add_argument('--max-audio-seconds', type=int, default=60)
        else:
            parser.add_argument('--text', required=True)
            parser.add_argument('--output', required=True)
            parser.add_argument('--max-frames', type=int, default=300)


def run_language(args):
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise ValueError('Language commands need PyTorch. Install the appropriate CPU/CUDA/ROCm torch build; see LANGUAGE.md.') from error
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    if args.action == 'doctor':
        from .backend import doctor
        doctor(args)
    elif args.action == 'train':
        from .training import train
        train(args)
    else:
        from .runtime import listen, speak, stream
        {'stream': stream, 'chat': stream, 'listen': listen, 'speak': speak}[args.action](args)
